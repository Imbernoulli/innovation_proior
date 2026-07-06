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
seeing each row alone.

Before I reach for a specific class, I want to be honest about the alternatives, because "instrument-independent"
does not uniquely pick one. There are really three candidates on the table given the frozen substrate.
The first is a richer linear/factor model — Ridge with cross-terms, or a shallow per-stock OLS — but any
linear response, even with hand-built interactions, hits the wall above: on 360 alphas I cannot enumerate
the order-two and order-three interactions that matter without guessing, and guessing wrong on a
360-wide table is how a factor model stays flat. The second is a sequence encoder — reshape each
`(stock, day)` to its `[60, 6]` window and run a recurrent net over the stock's own history. That is also
instrument-independent, and it does capture temporal structure the flat table discards. But I deliberately
do *not* want it as the floor, for two concrete reasons. One is methodological: a recurrent encoder is
exactly the machinery the relation-aware rungs will need to turn a stock into a vector before any
cross-stock mixing, so if I burn it here the floor and the graph model share a backbone and I can no
longer cleanly attribute a lift to "cross-section" versus "better single-stock encoding." The floor
should isolate the *one* thing I am about to add — relations — by being as unlike the graph model as an
honest strong baseline can be. The other is operational: a recurrent net trains stochastically, its
result wobbles with seed and initialization, and a floor that moves under me is a bad yardstick. The
third candidate is gradient-boosted decision trees on the flat 360-feature table, and it is the one I
fill the scaffold with, because on tabular data of this shape — on the order of half a million
`(stock, day)` rows for the csi300 training window, a few hundred dense engineered features — boosting
is both the strongest reliable hypothesis class and, being deterministic given the data and seed, a
fixed yardstick.

Let me be precise about why boosting is the right hypothesis class here and not just a reflex. A GBDT is
an additive ensemble of regression trees: at round `t` I hold the current predictor `F_{t-1}`, compute
for every instance the gradient of the loss against its current prediction, fit a new shallow tree to
the negative gradients, and fold it in shrunk by a learning rate. For the squared loss that I will use,
`L = ½(F − y)²`, the derivative with respect to the prediction is `∂L/∂F = F − y`, so `g_i = F_{t-1}(x_i) − y_i`
and `−g_i = y_i − F_{t-1}(x_i)` is exactly the residual — each tree fits what the ensemble is still
getting wrong. The trees are axis-aligned partitioners: each internal split is a threshold on a single
feature, and a leaf is a conjunction of such thresholds. That is precisely the "feature A matters only
when feature B is high" interaction the linear model could not express — a tree of depth `d` represents
an order-`d` interaction for free, and boosting stacks hundreds of them. And it does this without me
hand-specifying which interactions to look for, which on 360 alphas I could not do anyway. So the
boosting forest is the natural ceiling for an instrument-independent model on this table.

The data shape is what makes the *engine* matter, not just the model class. The split criterion at a
node is the variance gain in sum-of-gradient form,
`V_j(d) = (1/n)[ (Σ_{x_ij≤d} g_i)² / n_l(d) + (Σ_{x_ij>d} g_i)² / n_r(d) ]`, and evaluating it for every
feature and every candidate threshold is where the time goes — per tree the work scales like
`#data × #feature`. The pre-sorted exact method carries a sorted permutation of every feature around and
is memory-brutal: 360 sorted index arrays over half a million rows, plus the cost of a sort per feature
per node. The histogram method is the one I want underneath. It buckets each continuous feature into a
small fixed number of bins — 255, so a bin index fits in a byte — and for a node makes one pass
accumulating, per bin, the sum of gradients and the count. Building the histogram is `O(#data × #feature)`;
searching the `≤ #bin−1` boundaries is a rounding error next to it. A parent node's histogram is the sum
of its two children's, so the engine builds only the smaller child and recovers the sibling by
subtraction in `O(#bin)`. The engine grows leaf-wise — splitting the single leaf with the largest loss
reduction rather than a whole depth level — which reaches lower training loss for a fixed leaf budget,
capped by `num_leaves` and `max_depth` to bound overfit.

Let me put an order of magnitude on this, because "cheap enough" has to survive the harness budget and I
want to know it, not assume it. The root histogram over the training cross-section costs roughly
`#data × #features_sampled` gradient adds; with `colsample_bytree=0.8879` each tree sees about
`0.8879 × 360 ≈ 320` features, and with `subsample=0.8789` about `0.8789 × 500,000 ≈ 440,000` rows, so
the root pass is `≈ 440,000 × 320 ≈ 1.4×10⁸` accumulations. Deeper levels partition the data, so each
level costs no more than the one above, and the parent-minus-child subtraction shaves the larger sibling —
a single tree is a small multiple of that root pass, call it a few times `10⁸`. With up to `num_boost_round=1000`
trees that is on the order of `10¹¹`–`10¹²` operations for the whole forest, and at `num_threads=20`
processing hundreds of millions to low billions of simple integer/float adds per second per core, this
is a job measured in minutes, not hours. That is the practical difference between a usable floor and one
that times out, and it is why I want the histogram, leaf-wise engine specifically, with the
second-order/regularized gain (per-leaf `G=Σg`, `H=Σh`, leaf value `−G/(H+λ₂)` and L1 as a
soft-threshold on `G`) that ties leaf values to a regularized objective — exactly what the LightGBM
engine exposes.

Now the load-bearing part for *this task*: I must match what the harness actually exposes, not the
generic library default. The edit replaces `CustomModel` with a faithful wrapper around qlib's official
`LGBModel` (`gbdt.py`) and pulls the benchmark hyperparameters from qlib's
`workflow_config_lightgbm_Alpha360.yaml`. Two things about that are not the textbook defaults and I have
to respect them. First, the regularization is *enormous* and not optional: `lambda_l1 = 205.70`,
`lambda_l2 = 580.98`, `num_leaves = 210`, `max_depth = 8`, `colsample_bytree = 0.8879`,
`subsample = 0.8789`, `learning_rate = 0.0421`, with `num_boost_round = 1000` and early stopping at 50
rounds on the validation segment. Those L1/L2 values look absurd by general-tabular standards, so I want
to actually check what they do to a leaf rather than take them on faith. For squared loss the hessian is
`h_i = ∂²L/∂F² = 1`, so a leaf holding `n` samples has `H = n`, and its optimal value ignoring L1 is
`−G/(n + λ₂) = −G/(n + 580.98)`. Against the unregularized `−G/n` that is a shrink factor of
`n/(n + 580.98)`: a 100-sample leaf keeps only `100/681 ≈ 0.15` of its raw value, a 500-sample leaf
`500/1081 ≈ 0.46`, a 2000-sample leaf `0.78`, an 8000-sample leaf `0.93`. So `λ₂` does not shrink
uniformly — it shrinks *small* leaves toward zero and barely touches large ones, which is precisely the
behavior I want when the signal is faint: the forest is pushed to trust only large, well-populated leaves
and to distrust the small ones that would be fitting a handful of noisy rows. The L1 term is even more
telling. With soft-thresholding the leaf value is `−sign(G)·max(|G| − λ₁, 0)/(H + λ₂)`, so a leaf whose
`|G| = |Σ residual|` does not clear `λ₁ = 205.7` outputs *exactly zero* — it is pruned to a no-op. The
labels are `CSRankNorm`'d to roughly unit scale per day, so early residuals are order one and, once the
forest is finding a faint edge, a leaf's mean residual might be order `0.03`–`0.10`; then `|G| ≈ n × |mean residual|`,
and clearing `205.7` needs `n` on the order of two thousand to seven thousand samples. In other words a
leaf that captures only a faint edge over fewer than a couple thousand rows contributes nothing at all.
That is the mechanism by which 210 leaves per tree do not overfit: the vast majority of candidate leaves
are soft-thresholded to zero, and the *effective* capacity of the forest is far below its nominal leaf
count. These values are not absurd; they are the shape of a model that has been told, correctly, that
the cross-sectional signal here is small and easily faked. I take them verbatim.

The `num_leaves` and `max_depth` caps also want reading together rather than as two independent knobs. A
fully balanced depth-8 tree has `2⁸ = 256` leaves, and `num_leaves = 210` is `82%` of that — so the depth
cap and the leaf cap sit close to each other, and a tree is allowed to grow nearly to full depth-eight
resolution while leaf-wise growth spends its budget on the highest-loss regions rather than balancing.
The order of feature interaction the forest can express is therefore bounded at eight, which is generous
for a table whose real interactions are probably order two to four, and the leaf-wise budget makes sure
the deep branches only appear where they buy loss. The `learning_rate` of `0.0421` with up to 1000 rounds
and early stopping is the slow-and-many-trees regime that boosting needs when each tree can only chip a
little signal off the residual; with a rate this small the forest will keep adding trees until the
validation curve flattens and the 50-round patience trips, and the effective number of contributing
trees is set by the data, not by me.

Second — and this is the subtle harness detail — the LightGBM rung also has to *reset the dataset
preprocessing*, not just the model. The default `workflow_config.yaml` carries the neural-model handler:
features get `RobustZScoreNorm` + `Fillna` at inference, and the label gets `CSRankNorm` at training. The
inference-time `RobustZScoreNorm`/`Fillna` were put there for the graph models that need clean, normalized
tensors; they do not belong in the LightGBM baseline. I want to verify that claim rather than assert it,
because "trees don't care about scale" is the kind of thing that is *almost* true and then bites. A tree
split is a threshold test `x_j ≤ t`; under any strictly increasing transform `φ`, the test
`x_j ≤ t ⟺ φ(x_j) ≤ φ(t)` partitions exactly the same rows, and the gain depends only on how the
gradients partition, not on the feature values themselves — so per-feature standardization is genuinely a
no-op for the tree. But `RobustZScoreNorm` is not purely monotone: after subtracting the median and
dividing by the MAD it *clips* to a bounded range, and clipping is where information dies. Every feature
value beyond the clip is collapsed to the same boundary number, so the tree can no longer separate the
99th from the 99.9th percentile of, say, a volume spike — and extreme feature values are often the most
informative rows for next-day return. So blanking the inference processors is not cosmetic tidiness; it
actively preserves the tail separability that returns prediction leans on. The same logic condemns
`Fillna`: LightGBM routes NaN natively, learning at each split which direction missing rows should go,
which carries the information that a value *was* missing; `Fillna(0)` would overwrite that signal with a
literal zero that lands in the middle of a standardized distribution and looks like an ordinary,
uninformative observation. So the edit sets `infer_processors: []` and keeps only `DropnaLabel` +
`CSRankNorm` on the label.

The `CSRankNorm` on the label is the one piece I keep deliberately, and it is worth being clear about why
it survives when the feature processors do not. It rank-normalizes each day's labels cross-sectionally,
so the target the trees fit is not the raw next-day return but the stock's *rank* within that day's
cross-section, mapped to roughly unit scale. That is exactly right for a task scored by IC and Rank IC:
minimizing MSE against a per-day rank-normal target pushes the predictor's conditional mean to track the
within-day ordering, and IC (the Pearson correlation of prediction against label) and Rank IC are both
maximized when the prediction is monotone in that rank target — so an MSE objective on rank-normalized
labels is, for ranking purposes, aligned with the metric. It also stabilizes training across regimes: a
2008 crash day and a calm 2013 day both present the model with a comparable, standardized label
distribution, so no single volatile day dominates the gradient. The training data is pulled with the
learn key (`DK_L`), the validation set drives early stopping, and prediction maps the test features
through the learned trees and returns a `pd.Series` indexed by `(datetime, instrument)`, which is the
contract the `SignalRecord`/`PortAnaRecord` chain consumes. There is no per-day batching here and no
concept matrix lookup — the whole point of this rung is that it never opens the graph file.

Let me sanity-check the whole configuration against its own limits before I trust it. If I sent `λ₂ → ∞`
every leaf value goes to zero, the forest predicts a constant, and IC collapses to zero — a degenerate
but reassuring endpoint, because it tells me the regularization is a dial between "predict nothing" and
"overfit," and `580.98` is a finite interior point chosen so that only well-supported leaves fire. If I
sent `λ₂ → 0` and `λ₁ → 0` the 210-leaf trees would happily carve the 2008–2014 window into thousands of
tiny high-variance leaves and generalize to nothing on 2017–2020 — the standard boosting failure on
low-signal data. The published settings sit deliberately toward the "predict nothing" end of that dial,
which is the correct bias for a signal this faint. And the row bagging is worth naming precisely: with the
default `gbdt` booster the `subsample = 0.8789` is ordinary stochastic bagging (the GOSS machinery that
weights large-gradient rows is *not* switched on — the engine forwards `subsample` to plain bagging), so
each tree sees a different ~440,000-row sample of ~320 features, which decorrelates the trees and is the
one remaining source of variance reduction on top of the shrinkage.

One thing I should be clear-eyed about, because it is the limitation the next rung exists to attack:
none of this machinery — histogram binning, second-order gain, heavy shrinkage, row bagging — changes the
fundamental fact that every row is an independent example. Two rows on the same trading day, two stocks in
the same sector reacting to the same macro print, are to this model two unrelated points in a
360-dimensional space. The trees can find that "high-momentum-plus-low-volatility" rows tend to
outperform, but they cannot find that "this stock moves with that stock today." The cross-section — the
one structural fact the research question is built on — is invisible. That is the deliberate ceiling of
this rung.

So I run it and I form falsifiable expectations against what the relation-aware rungs will have to clear.
This is the strongest *instrument-independent* model, so it should be a real floor, not a trivial one:
on the main csi300 universe I expect positive but modest signal — an IC in the low tens of a percent and
an ICIR comfortably positive, since rank-normalized boosting on Alpha360 is a known-decent factor model.
But I expect its *portfolio* numbers to be the weakest of the ladder: the TopkDropout backtest cares
about getting the top-50 ordering right day after day, and a row-independent model that cannot use
co-movement should produce a noisier daily ranking, so its information ratio should be low — far below
what a model that reads the cross-section can reach — and its annualized return small. On the harder,
lower-breadth csi100 universe, where there are fewer names to diversify the noise and the cross-sectional
signal is thinner, I expect it to struggle most, possibly with a negative portfolio return, because with
only a hundred names the idiosyncratic noise in the daily top-50 has far less breadth to average against.
Because the GBDT is deterministic given the data and seed, I also expect its three-seed rows to be
identical — which, if it holds, is a useful signal in itself: any variance I later see in a neural rung is
structural, not the floor moving under me. If the graph rungs cannot beat *this* — if reading the day's
cross-section does not lift IC and especially information ratio above the boosting floor — then the entire
premise that cross-stock structure is exploitable is in doubt. That is the bar this rung sets, and it is
exactly the gap I will reach for next: turn off the row-independence assumption by letting each day's
stocks influence each other's predictions.
