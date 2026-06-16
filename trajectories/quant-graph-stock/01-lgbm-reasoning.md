The relation-aware model is the whole point, but it has to beat *something*, and the honest floor is
the strongest predictor that uses no cross-stock structure at all. So before I touch a graph, I want
the best instrument-independent model I can run on this exact harness — if a model that treats every
`(stock, day)` row in isolation already captures most of the predictable signal, then whatever a graph
buys me has to be measured against that, not against the trivial Ridge the scaffold ships with. The
scaffold default is a single linear weight vector over the 360 Alpha360 features; it is stable and it
will run, but a global linear response is a weak hypothesis for a feature table this wide. The features
are engineered alphas — ratios, momentum windows, volatility proxies — and the relationship between any
one of them and next-day return is not linear, and certainly not additive across features. A momentum
feature matters only in some volatility regime; a volume spike means one thing after a run-up and another
after a drawdown. Linear regression cannot express "feature A matters only when feature B is high." So
the right floor is a model that captures non-linear feature interactions while still, deliberately,
seeing each row alone. On tabular data of this shape — millions of `(stock, day)` rows, a few hundred
dense engineered features — the consensus answer is gradient-boosted decision trees, and that is what I
fill the scaffold with at this rung.

Let me be precise about why boosting is the right hypothesis class here and not just a reflex. A GBDT is
an additive ensemble of regression trees: at round `t` I hold the current predictor `F_{t-1}`, compute
for every instance the gradient of the loss against its current prediction, fit a new shallow tree to
the negative gradients, and fold it in shrunk by a learning rate. For the squared loss that I will use,
the gradient is `g_i = F_{t-1}(x_i) - y_i`, so `-g_i` is exactly the residual and each tree fits what
the ensemble is still getting wrong. The trees are axis-aligned partitioners: each internal split is a
threshold on a single feature, and a leaf is a conjunction of such thresholds. That is precisely the
"feature A matters only when feature B is high" interaction the linear model could not express — a tree
of depth `d` represents an order-`d` interaction for free, and boosting stacks hundreds of them. And it
does this without me hand-specifying which interactions to look for, which on 360 alphas I could not do
anyway. So the boosting forest is the natural ceiling for an instrument-independent model on this table.

The data shape is what makes the *engine* matter, not just the model class. The split criterion at a
node is the variance gain in sum-of-gradient form,
`V_j(d) = (1/n)[ (Σ_{x_ij≤d} g_i)² / n_l(d) + (Σ_{x_ij>d} g_i)² / n_r(d) ]`, and evaluating it for every
feature and every candidate threshold is where the time goes — per tree the work scales like
`#data × #feature`. The pre-sorted exact method carries sorted permutations around and is memory-brutal;
the histogram method is the one I want underneath. It buckets each continuous feature into a small fixed
number of bins — 255, so a bin index fits in a byte — and for a node makes one pass accumulating, per
bin, the sum of gradients and the count. Building the histogram is `O(#data × #feature)`; searching the
`≤ #bin−1` boundaries is a rounding error next to it. A parent node's histogram is the sum of its two
children's, so the engine builds only the smaller child and recovers the sibling by subtraction in
`O(#bin)`. The engine grows leaf-wise — splitting the single leaf with the largest loss reduction rather
than a whole depth level — which reaches lower training loss for a fixed leaf budget, capped by
`num_leaves` and `max_depth` to bound overfit. These are the properties that make the boosting forest
not just accurate but *cheap enough* to train to convergence on the full 2008–2014 training cross-section
in the harness's time budget, which on this many rows is the practical difference between a usable
baseline and one that times out. So I want the histogram-based, leaf-wise GBDT engine specifically, with
the second-order/regularized gain (per-leaf `G=Σg`, `H=Σh`, leaf value `-G/(H+λ₂)` and L1 as a
soft-threshold on `G`) that ties leaf values to a regularized objective — exactly what the LightGBM
engine exposes.

Now the load-bearing part for *this task*: I must match what the harness actually exposes, not the
generic library default. The edit replaces `CustomModel` with a faithful wrapper around qlib's official
`LGBModel` (`gbdt.py`) and pulls the benchmark hyperparameters from qlib's
`workflow_config_lightgbm_Alpha360.yaml`. Two things about that are not the textbook defaults and I have
to respect them. First, the regularization is *enormous* and not optional: `lambda_l1 = 205.70`,
`lambda_l2 = 580.98`, `num_leaves = 210`, `max_depth = 8`, `colsample_bytree = 0.8879`,
`subsample = 0.8789`, `learning_rate = 0.0421`, with `num_boost_round = 1000` and early stopping at 50
rounds on the validation segment. Those L1/L2 values look absurd by general-tabular standards, but they
are the published Alpha360 settings, and they exist because financial cross-sectional signal is faint and
noisy: the signal-to-noise ratio is so low that without heavy shrinkage the 210-leaf trees would happily
fit microstructure noise in the 2008–2014 window and generalize to nothing on 2017–2020. The big `λ`
values push most candidate splits' gains below the complexity charge, so the forest stays shallow in
*effective* capacity even at 210 leaves. The learning rate of 0.0421 with up to 1000 rounds and early
stopping is the slow-and-many-trees regime that boosting needs when each tree can only chip a little
signal off the residual. I take these verbatim; they are the difference between this baseline being a
fair floor and being a broken one.

Second — and this is the subtle harness detail — the LightGBM rung also has to *reset the dataset
preprocessing*, not just the model. The default `workflow_config.yaml` carries the neural-model handler:
features get `RobustZScoreNorm` + `Fillna` at inference, and the label gets `CSRankNorm` at training. The
inference-time `RobustZScoreNorm`/`Fillna` were put there for the graph models that need clean, normalized
tensors; they do not belong in the LightGBM baseline, which feeds raw feature values to a tree engine
that is scale-invariant by construction (a tree split is a threshold; monotone rescaling of a feature
does not change which rows fall on each side, so per-feature normalization is a no-op for trees and only
risks discarding information through the outlier clipping). So the edit blanks the inference processors
(`infer_processors: []`) and keeps only `DropnaLabel` + `CSRankNorm` on the label. The `CSRankNorm` on the
label is the one piece I keep deliberately: it rank-normalizes each day's labels cross-sectionally, which
is exactly right for a ranking objective scored by IC and Rank IC — the model learns to order stocks
within a day rather than to hit absolute return levels. The training data is pulled with the learn key
(`DK_L`), the validation set drives early stopping, and prediction maps the test features through the
learned trees and returns a `pd.Series` indexed by `(datetime, instrument)`, which is the contract the
SignalRecord/PortAnaRecord chain consumes. There is no per-day batching here and no concept matrix
lookup — the whole point of this rung is that it never opens the graph file. The full scaffold fill is in
the answer.

One thing I should be clear-eyed about, because it is the limitation the next rung exists to attack:
this engine does GOSS and EFB and second-order gain, but with the default GBDT booster and a `subsample`
of 0.8789 it is just doing ordinary stochastic row bagging — the GOSS machinery is not switched on, the
engine forwards `subsample` to bagging — and *none* of it changes the fundamental fact that every row is
an independent example. Two rows on the same trading day, two stocks in the same sector reacting to the
same macro print, are to this model two unrelated points in a 360-dimensional space. The trees can find
that "high-momentum-plus-low-volatility" rows tend to outperform, but they cannot find that "this stock
moves with that stock today." The cross-section — the one structural fact the research question is built
on — is invisible. That is the deliberate ceiling of this rung.

So I run it and I form falsifiable expectations against what the relation-aware rungs will have to clear.
This is the strongest *instrument-independent* model, so it should be a real floor, not a trivial one:
on the main csi300 universe I expect positive but modest signal — an IC in the low tens of a percent and
an ICIR comfortably positive, since rank-normalized boosting on Alpha360 is a known-decent factor model.
But I expect its *portfolio* numbers to be the weakest of the ladder: the TopkDropout backtest cares
about getting the top-50 ordering right day after day, and a row-independent model that cannot use
co-movement should produce a noisier daily ranking, so its information ratio should be low — far below
what an attention-over-the-cross-section model can reach — and its annualized return small. On the
harder, lower-breadth csi100 universe, where there are fewer names to diversify the noise and the
cross-sectional signal is thinner, I expect it to struggle most, possibly with a negative portfolio
return. If the graph rungs cannot beat *this* — if attention over the day's stocks or concept-aware
aggregation does not lift IC and especially information ratio above the boosting floor — then the entire
premise that cross-stock structure is exploitable is in doubt. That is the bar this rung sets, and it is
exactly the gap I will reach for next: turn off the row-independence assumption by letting each day's
stocks attend to each other.
