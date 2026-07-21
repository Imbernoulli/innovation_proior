The Transformer came back exactly the way I feared, and the numbers say *which* failure it was. On csi300
the IC is 0.0117 and the Rank IC 0.0350 — a quarter of what a working temporal model reaches — and the
portfolio information ratio is **−0.44**, with annualized return **−0.036**. On csi100 it is worse: IR
−1.01, return −0.059. On csi300_recent the IC is essentially zero at 0.0014, IR −1.09. A negative IR
across all three universes is the tell I called in advance: a near-noise ranking is still forced through
TopkDropout to hold fifty names and churn five a day, so it pays transaction costs on noise and the return
goes negative. The signal never formed.

Three things in these numbers carry more than "it failed." First, the IC-to-Rank-IC gap: csi300 shows IC
0.0117 but Rank IC 0.0350, a factor of three. Rank IC being the larger means whatever faint ordering
exists is more monotone-in-rank than linear — the Pearson IC is being dragged toward zero by a handful of
large-magnitude predictions that are directionally not crazy but numerically off, the fingerprint of an
under-trained regressor whose outputs are uncalibrated in scale. Second, I can back out the strategy
volatility, since IR is annualized return over annualized active risk: on csi300 `0.036 / 0.44 ≈ 0.082`,
on csi100 `0.059 / 1.01 ≈ 0.058`, on csi300_recent `0.085 / 1.09 ≈ 0.078`. So the strategy runs at roughly
6–8% active volatility on all three and simply points the wrong way — the bets are not wild, a near-random
ranking just bleeds turnover cost steadily. Third, and this is the observation I carry forward: csi100's
IC (0.0204) is *higher* than csi300's (0.0117), yet csi100's IR (−1.01) is more than twice as negative as
csi300's (−0.44). Higher signal, worse portfolio. That decoupling of IC from IR on the smaller universe
says the portfolio construction on csi100 punishes a given ranking quality harder — a hint I should not
forget — but for *this* attempt the dominant fact is simpler: every universe's signal is near noise
because the optimization never caught.

And I know why: the brittle-transformer story — constant `1e-4` Adam with no warmup, patience-5 early
stop, on a faint signal — almost certainly early-stopped on a barely-trained model. The architecture's
bias was fine; the *optimization* was the problem, and the fixed training loop gives the most
init-sensitive model the least protection. So the lesson is not "add more architecture." It is the
opposite: I want a learner that needs no delicate optimization at all — no schedule to thread, no
initialization to survive, no early-stop gamble — robust on noisy tabular data out of the box.

The tempting move, given "the failure was optimization, not bias," is to *keep* the transformer and fix
its training: the editable `CustomModel` does let me write a manual warmup-then-decay schedule inside the
loop. I could size it — with ~250 steps per epoch, a warmup of four thousand steps is about sixteen epochs
of ramp before the rate even reaches `1e-4`. But that is the move I should *not* make, and the reason is
in the numbers. The transformer's problem is not a single missing schedule; it is a *stack* of
fragilities — init sensitivity, a 94%-FFN parameter mass with dropout off, patience-5 cutting the run
short, and a faint signal giving the optimizer almost no gradient to lean on. Warmup addresses one while
leaving me on the same high-wire, and a sixteen-epoch warmup fights directly against a patience-5 stop
that can terminate before the ramp even finishes. Just lowering the rate or widening the patience is
guessing hyperparameters on a model whose whole character is that it is hard to train. The directed move
is to change the *class* of learner so the entire family of optimization fragilities disappears at once —
away from sequence models, back to the consensus high-accuracy learner on structured data: a
gradient-boosted decision tree.

A GBDT is an additive sequence of regression trees. At round `t` I have the current ensemble `F_{t-1}`,
compute for every instance the gradient of the loss against its prediction, `g_i = ∂L(y_i, F_{t-1}(x_i))
/∂F`, fit a new tree to the negative gradients, and fold it in shrunk by a learning rate. For squared
loss `g_i = ŷ_i − y_i`, so each tree fits the residual — the classic residual-fitting form. There is no
global initialization that can poison the whole run and no single learning rate whose miscalibration
kills convergence: shrinkage is a per-tree scalar that only slows things down safely, early stopping is on
the number of *trees* (a discrete, monotone count) rather than a gamble on a noisy validation curve, and
the trees discover nonlinear feature interactions automatically. Against a failure that was pure training
fragility, swapping in a learner with essentially no optimization fragility is the directed fix.

Within the tree family, why boosting rather than a random forest, which is also robust and
initialization-free? A forest grows many *independent* deep trees on bootstrap samples and averages them;
its variance reduction comes from decorrelating high-variance, low-bias trees. Boosting grows *dependent*
shallow-ish trees, each fitting what the ensemble still gets wrong, with a shrinkage rate that lets me
take the additive fit arbitrarily slowly. The decisive difference is the control surface. A forest has no
learning rate — each tree contributes a full `1/T` share and I cannot dial the aggressiveness down toward
a faint signal; my only knobs are tree count and depth. Boosting hands me `learning_rate` and a per-tree
early stop, so I can make the ensemble crawl toward the signal and halt the instant validation stops
improving — exactly the "act only on strong, repeated signal" behavior a low-IC problem demands. And the
forest's own strength, deep high-variance trees, is a liability here: on a signal this faint they mostly
fit noise that averaging only partly cancels, whereas heavily shrunk boosting never lets any single tree
commit hard to noise in the first place. So boosting, not bagging. (The residual picture also squares with
the label pipeline: `CSRankNorm` makes each cross-section's targets mean-zero, so the first tree's
constant leaf is `mean(y) ≈ 0` and the ensemble does nothing until a split isolates a subpopulation whose
mean return departs from zero.)

What the trees see is the one place this differs from a naive "just use a tree." The transformer treated
the 360 features as a 60×6 sequence; a tree treats them as a flat tabular row, six base ratios at sixty
lags, each a column. The tree is blind to the fact that column 5 and column 65 are the same ratio one day
apart — but on this data that blindness may not hurt much, because the engineered Alpha360 ratios already
carry the relevant information per-column, and the tree's job is to find which lags and which interactions
predict the forward return. A split on "the 1-day return is high AND the 20-day volume ratio is low" is
exactly the kind of nonlinear factor interaction a hand-built factor model would have to specify by hand.
So the loss of the temporal bias is real, but the gain in robustness and automatic interaction discovery
is, I expect, the better trade against a near-zero-signal baseline.

Now the engine. Growing each tree is dominated by split finding: to split a node I need, per feature, the
gain of every candidate split, which scans the node's data — `O(#data × #feature)` per tree. On the same
`~5·10⁵` stock-days and 360 features, a single node's exact split search is on the order of `1.9·10⁸`
operations. The histogram engine cuts this: bucket each continuous feature into a fixed number of bins
(255, so a bin index fits in a byte), and for a node make one pass accumulating per-bin sum-of-gradients
and count. Searching splits is then `O(#bin × #feature) = 255·360 ≈ 9.2·10⁴`, three orders below the
exact scan. The binning also settles the input scaling: bin edges are chosen once from the feature's
empirical distribution — quantile-style, each bin holding roughly equal data mass — so a monotone
rescaling of a feature leaves its quantiles, bin assignments, and every candidate split unchanged. That
is why the trees need no standardization, and the mechanical reason the `RobustZScoreNorm` the neural
model depended on is not merely unnecessary here but, in its clipping, actively harmful (I return to this
at the processor edit). A parent's histogram is the sum of its two children's, so I build only the smaller
child and recover the sibling by subtraction in `O(#bin)`. The engine grows leaf-wise — best-first,
splitting the leaf with the largest loss reduction — which reaches lower training loss than level-wise at
a fixed leaf budget, capped by `num_leaves` and `max_depth` to bound overfit.

The split criterion is the variance gain in sum-of-gradient form, which a second-order view generalizes to
leaf value `w* = −G/(H+λ2)` with `G = Σg`, `H = Σh`, L2 penalty `λ2`, and L1 entering as a soft-threshold
on `G`; for squared loss the Hessian is 1 and this reduces to the plain variance gain. The two famous
accelerations — GOSS, keeping large-gradient rows whole and reweighting the small-gradient tail, and EFB,
packing mutually exclusive sparse features into one column — are why the method is fast, but I should be
honest about how much fires *here*. This runs the default `gbdt` booster, not `goss`, so
`subsample`/`bagging_fraction` is just plain uniform row subsampling (the `a=0` degenerate case of GOSS),
not the importance sampler; and EFB is largely inert because Alpha360 features are dense continuous ratios,
not sparse and exclusive. The accuracy and robustness here come from the histogram-leaf-wise engine and
the regularization, not from GOSS/EFB — I want that clear so I do not import the big-data speed story into
a setting where it mostly does not bite.

The hyperparameters do real regularization work. `learning_rate = 0.0421` is heavy shrinkage — each of up
to `num_boost_round = 1000` trees nudges the prediction gently and the ensemble averages out noise rather
than chasing it; early stopping at patience 50 cuts the round count before it overfits. `num_leaves = 210`
with `max_depth = 8` allows fairly expressive trees — note `2⁸ = 256 > 210`, so `max_depth` binds the
deepest paths while `num_leaves` caps the total, letting the tree be unbalanced without exploding. The two
L-penalties are enormous — `lambda_l1 = 205.7`, `lambda_l2 = 580.98` — and characteristic of financial
data, so it is worth working out what they do to a leaf via `w* = −sign(G)·max(|G| − λ1, 0) / (H + λ2)`.
The L1 term is a hard gate: a leaf's gradient sum `G` must exceed `205.7` in magnitude before the leaf
gets *any* nonzero value. With rank-normalized targets whose per-sample gradient is `O(1)`, that means a
leaf needs on the order of a couple hundred samples all pulling their residual the same direction — pure
noise, whose signs cancel, never clears the bar. The L2 term then shrinks whatever survives: for a leaf of
a thousand samples, `H ≈ 1000`, so the denominator is `1000 + 581 = 1581` and the leaf value is scaled to
`1000/1581 ≈ 0.63` of its unpenalized size, a further 37% haircut on top of the L1 gate and the `0.0421`
rate. Stack the three and each tree moves the ensemble a genuinely tiny, noise-averaged step. That is why
the nominal capacity — up to a thousand trees of up to 210 leaves, order `2·10⁵` leaf values grown
*adaptively*, normally a recipe for overfit — does not bite: the L1 gate zeros most candidate leaves, the
L2 shrinkage and small rate scale down the survivors, early stopping caps the round count well below a
thousand, and each surviving leaf is an average over hundreds of samples rather than a free weight. The
effective degrees of freedom are a tiny fraction of the nominal count — precisely what the brittle
transformer, with its 94%-FFN mass and dropout off, did *not* have. `colsample_bytree = 0.8879` and
`subsample = 0.8789` add column- and row-level stochasticity so successive trees decorrelate; `objective
= mse` matches the label. Every knob is pushed toward *more* regularization than a default tabular task
would use, because the IC ceiling is low and the failure mode is variance, not bias.

There is one task-specific edit, and it is the kind of same-named-but-different detail I have to get right:
this does *not* leave the workflow processors alone. The default applies `RobustZScoreNorm` +
`Fillna` as `infer_processors`, there for the gradient-descent models that need standardized inputs to
train stably. A tree does not — it splits on order statistics, so monotone rescaling is irrelevant. But
`RobustZScoreNorm` *clips* to roughly `±3` robust standard deviations before standardizing, and clipping
is exactly the non-monotone operation that hurts a tree: every value beyond the boundary is flattened to
the same number, so the tree can no longer order or split *among* the extremes — and in return prediction
the extreme moves are often the informative ones. So the edit sets `infer_processors: []` — raw features to
the trees — keeping only the label-side `DropnaLabel` + `CSRankNorm`. Dropping the block also drops
`Fillna`, which is safe: the histogram engine handles missing values natively, holding them out of the
binned histogram and, at each split, learning which child the NaNs default to by trying both and keeping
the higher-gain side. So the tree treats "this ratio is missing" as its own routable signal rather than
imputing it to a fake number — strictly better than `Fillna`'s zero-fill. The neural model, by contrast,
cannot feed a NaN through a matrix multiply, which is why `Fillna` is non-negotiable for it and discardable
here. The full module and the processor edit are in the answer.

So the change from the Transformer is total: throw out the sequence model and its delicate Adam training,
treat the 60×6 window as a flat tabular row, and fit a heavily-regularized histogram GBDT that needs no
warmup, no initialization luck, and no early-stop gamble — while stripping the neural-only feature
normalization so the trees see the raw signal.

What is falsifiable against the transformer's numbers? First and most important, the IR must go *positive*
on csi300 — the transformer's −0.44 was a near-noise ranking paying churn costs, and a GBDT that actually
forms signal should rank well enough to earn back those costs; if the csi300 IR comes back negative, the
robustness argument is wrong. Second, the IC should climb substantially from 0.0117 toward the
low-single-digit-percent range — anything still near 0.01 would mean the loss of the temporal bias cost
more than the robustness gained. Third, csi100 is where I am genuinely uncertain: its transformer IC
(0.020) outran csi300's (0.012) yet its return was the most negative, and the IC-to-IR decoupling I
traced says ranking quality does not translate into portfolio return there the way it does on csi300,
because fifty of a hundred names is a much larger slice and the TopkDropout churn bites harder. So I
expect a clear positive csi300 IR and a real IC jump, but would not be surprised if csi100's return stays
weak — even negative — even as its IC improves. If so, the diagnosis for the next attempt is that the
tree's per-row tabular view, however robust, still leaves ranking quality on the table that a temporal
model trained *robustly* — not the brittle transformer — could recover.
