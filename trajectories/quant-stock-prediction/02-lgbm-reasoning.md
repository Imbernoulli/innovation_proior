The Transformer rung came back exactly the way I feared, and the numbers say *which* failure it was.
On csi300 the IC is 0.0117 and the Rank IC 0.0350 — the IC is a quarter of what a working temporal model
reaches, and crucially the portfolio information ratio is **−0.44**, with annualized return **−0.036**.
On csi100 it is worse: IR −1.01, return −0.059. On csi300_recent, IC essentially zero at 0.0014 and IR
−1.09. A negative IR across all three universes is the tell I called in advance: a model whose ranking is
near-noise still gets forced through TopkDropout to hold fifty names and churn five a day, so it pays
transaction costs on noise and the return goes negative. The signal never formed. And I know why — the
brittle-transformer story I laid out: a constant `1e-4` Adam with no warmup, patience-5 early stop, on a
faint signal, almost certainly early-stopped on a barely-trained model. The architecture's bias was
fine; the *optimization* was the problem, and the harness gives the most init-sensitive model the least
protective training loop. So the lesson is not "add more architecture." It is the opposite: I want a
learner that needs no delicate optimization at all — no learning-rate schedule to thread, no
initialization to survive, no early-stop gamble — and that is robust on noisy tabular data out of the box.
That points away from sequence models entirely, back to the consensus high-accuracy learner on
structured data: a gradient-boosted decision tree.

Let me be precise about why a tree ensemble is the right move *here*, against this measured failure. A
GBDT is an additive sequence of regression trees. At round `t` I have the current ensemble `F_{t-1}`, I
compute for every instance the gradient of the loss against its current prediction, `g_i = ∂L(y_i,
F_{t-1}(x_i))/∂F`, fit a new tree to the negative gradients, and fold it in shrunk by a learning rate.
For squared loss `g_i = ŷ_i − y_i`, so each tree fits the residual `y_i − ŷ_i` — the classic
residual-fitting form. There is no global initialization that can poison the whole run, no single
learning rate whose miscalibration kills convergence: shrinkage is a per-tree scalar that only ever
slows things down safely, early stopping is on the number of *trees* (a discrete, monotone count) rather
than a gamble on a noisy validation curve, and the trees discover nonlinear feature interactions
automatically. On a low signal-to-noise regression where the transformer's training fragility was the
whole failure, swapping in a learner with essentially no optimization fragility is the directed fix.

But I should think about what the trees see, because that is the one place this rung differs from a naive
"just use a tree." The transformer treated the 360 features as a 60×6 sequence; a tree treats them as a
flat tabular row, six base ratios at sixty lags, each a column. The tree is blind to the fact that
column 5 and column 65 are the same ratio one day apart — but on this data that blindness may not hurt
much, because the engineered Alpha360 ratios already carry the relevant information per-column, and the
tree's job is to find which lags and which interactions of those columns predict the forward return. A
tree splitting on "the 1-day return is high AND the 20-day volume ratio is low" is exactly the kind of
nonlinear factor interaction a hand-built factor model would have to specify by hand. So the loss of the
temporal bias is real but the gain in robustness and automatic interaction discovery is, I expect, the
better trade against a near-zero-signal baseline.

Now the engine. The cost of growing each tree is dominated by split finding: to split a node I need, for
every feature, the gain of every candidate split, and that scans the node's data, so per-tree work is
`O(#data × #feature)`. The histogram engine cuts this to size: bucket each continuous feature into a
small fixed number of bins (255, so a bin index fits in a byte), and for a node make one pass
accumulating per-bin sum-of-gradients and count. Searching splits is then `O(#bin × #feature)`, a
rounding error next to the build. A parent node's histogram is the sum of its two children's, so I build
only the *smaller* child's histogram and recover the sibling by subtraction in `O(#bin)`. On top of that
the engine grows leaf-wise — best-first, splitting the single leaf with the largest loss reduction at
each step — which reaches lower training loss than level-wise at a fixed leaf budget, capped by
`num_leaves` and `max_depth` to bound overfit. These are the knobs the deployment exposes, and they are
what make a thousand-tree ensemble tractable on this data.

The split criterion is the variance gain in sum-of-gradient form,
`V_j(d) = (1/n)[ (Σ_{x_ij≤d} g_i)²/n_l(d) + (Σ_{x_ij>d} g_i)²/n_r(d) ]`,
which a second-order view generalizes to leaf value `w* = −G/(H+λ2)` with `G = Σg`, `H = Σh` and an L2
penalty `λ2`, L1 entering as a soft-threshold on `G`. For squared loss the Hessian is 1 and this reduces
to the plain variance gain. The two famous accelerations the engine carries — GOSS, which keeps the
large-gradient (under-trained) rows whole and samples the small-gradient tail reweighted by `(1−a)/b` to
keep the gain unbiased, and EFB, which packs mutually exclusive sparse features into one column by
disjoint bin-offset ranges via a greedy graph-coloring with a small conflict budget — are the reason the
method is *fast*. I should be honest about how much of that fires *here*, though, because the task's
deployment is specific: it runs the default `gbdt` booster, not `goss`, so the row-sampling reduction is
*not* the GOSS importance sampler — `subsample`/`bagging_fraction` just feeds the engine's plain bagging
machinery (uniform row subsampling per round, which is the `a=0` degenerate case of GOSS). EFB still
applies internally wherever the Alpha360 columns are sparse and exclusive, but Alpha360 features are
dense continuous ratios, so EFB's column packing is largely inert on this input. The accuracy and the
robustness come from the histogram-leaf-wise engine and the regularization, not from GOSS/EFB on this
particular dense feature matrix. I want that clear so I am not importing the big-data speed story into a
setting where it mostly does not bite.

Now the hyperparameters, which are the official qlib Alpha360 LightGBM benchmark and are doing real
regularization work that I should justify rather than copy. `learning_rate = 0.0421` is small —
heavy shrinkage, so each of up to `num_boost_round = 1000` trees nudges the prediction gently and the
ensemble averages out noise rather than chasing it; early stopping at patience 50 on the validation set
cuts off the round count before it overfits. `num_leaves = 210` with `max_depth = 8` allows fairly
expressive trees, but the two L-penalties are enormous — `lambda_l1 = 205.7`, `lambda_l2 = 580.98` — and
that is deliberate and characteristic of financial data: with such large penalties on the leaf weights,
the soft-threshold on `G` zeros out splits whose gradient mass is small, so the trees only act on the
strongest, most repeated signal and ignore the vast noise floor. `colsample_bytree = 0.8879` and
`subsample = 0.8789` (with the default bagging) add column- and row-level stochasticity so successive
trees decorrelate and the ensemble generalizes. `objective = mse` matches the regression label. Every one
of these is pushed toward *more* regularization than a default tabular task would use, because the IC
ceiling here is low and the failure mode is variance, not bias.

There is one more task-specific edit that matters, and it is exactly the kind of same-named-but-different
detail I have to get right: this rung does *not* leave the workflow processors alone. The default
workflow applies a neural-model preprocessing block — `RobustZScoreNorm` (clip-and-standardize the
features) and `Fillna` as `infer_processors`. Those are there for the gradient-descent models, which need
standardized inputs to train stably. A tree does not: it splits on order statistics, so monotone
per-feature rescaling is irrelevant to it, and the robust-z-score clipping actually *removes* information
the tree could split on (the clipped tails). So the lgbm edit sets `infer_processors: []` — feed the
trees the raw features — and keeps only the label-side `DropnaLabel` + `CSRankNorm`. This is a genuine
deviation from the transformer rung, which used the default processor block unchanged, and it is the
correct one: the preprocessing that *helped* the neural model would have quietly *hurt* the tree. (The
full scaffold module and the processor edit are in the answer.)

So the delta from the Transformer rung is total: I throw out the sequence model and its delicate Adam
training, treat the 60×6 window as a flat tabular row, and fit a heavily-regularized histogram GBDT that
needs no warmup, no initialization luck, and no early-stop gamble — while stripping the
neural-only feature normalization so the trees see the raw signal.

What do I expect against the Transformer's measured numbers, and what is the falsifiable claim? First and
most important, the IR must go *positive* on csi300 — the transformer's −0.44 was a near-noise ranking
paying churn costs, and a GBDT that actually forms signal should rank well enough to earn back the costs;
if the csi300 IR comes back negative, the tree failed to find signal and my robustness argument is wrong.
Second, the IC should climb substantially from 0.0117 toward the high-0.03s / low-0.04s a working model
on this data reaches — anything still near 0.01 would mean the loss of the temporal bias cost more than
the robustness gained. Third, where I am genuinely uncertain: on csi100, even the transformer's IC
(0.020) outran its csi300 IC, yet its return was deeply negative; csi100 is a smaller, harder universe
where *every* model on this ladder may struggle to earn a positive portfolio return even with a decent
IC, because fifty names out of a hundred is a much larger slice of the universe and the TopkDropout churn
bites harder. So my honest expectation is: a clear positive csi300 IR and a real IC jump (the robustness
fix working), but I would not be surprised if csi100's portfolio return stays weak even as its IC
improves — and if so, the diagnosis for the next rung is that the tree's per-row tabular view, however
robust, still leaves ranking quality on the table that a temporal model trained *robustly* (not the
brittle transformer) could recover. That is the fork the final rung takes.
