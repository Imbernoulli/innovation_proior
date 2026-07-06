The Transformer rung came back exactly the way I feared, and the numbers say *which* failure it was.
On csi300 the IC is 0.0117 and the Rank IC 0.0350 — the IC is a quarter of what a working temporal model
reaches, and crucially the portfolio information ratio is **−0.44**, with annualized return **−0.036**.
On csi100 it is worse: IR −1.01, return −0.059. On csi300_recent, IC essentially zero at 0.0014 and IR
−1.09. A negative IR across all three universes is the tell I called in advance: a model whose ranking is
near-noise still gets forced through TopkDropout to hold fifty names and churn five a day, so it pays
transaction costs on noise and the return goes negative. The signal never formed.

Let me read these numbers a little harder before I move, because they carry more than "it failed." First,
the IC-to-Rank-IC gap: csi300 shows IC 0.0117 but Rank IC 0.0350, a factor of three. Rank IC being the
larger of the two means whatever faint ordering exists is more monotone-in-rank than linear — the Pearson
IC is being dragged toward zero by a handful of large-magnitude predictions that are directionally not
crazy but numerically off, which is exactly the fingerprint of an under-trained regressor whose outputs
have not been calibrated in scale. Second, I can back out the strategy volatility from the portfolio
numbers, since the information ratio is annualized return over annualized active risk. On csi300,
`0.036 / 0.44 ≈ 0.082`; on csi100, `0.059 / 1.01 ≈ 0.058`; on csi300_recent, `0.085 / 1.09 ≈ 0.078`. So
the strategy is running at roughly 6–8% active volatility on all three and simply pointing the wrong way
— it is not that the bets are wild, it is that a near-random ranking bleeds the turnover cost steadily.
Third, and this is the observation I will carry forward: csi100's IC (0.0204) is actually *higher* than
csi300's (0.0117), yet csi100's IR (−1.01) is more than twice as negative as csi300's (−0.44). Higher
signal, worse portfolio. That decoupling of IC from IR on the smaller universe is a hint I should not
forget — it says the portfolio construction on csi100 punishes a given ranking quality harder than on
csi300 — but for *this* rung the dominant fact is simpler: every universe's signal is near noise because
the optimization never caught.

And I know why — the brittle-transformer story I laid out: a constant `1e-4` Adam with no warmup,
patience-5 early stop, on a faint signal, almost certainly early-stopped on a barely-trained model. The
architecture's bias was fine; the *optimization* was the problem, and the harness gives the most
init-sensitive model the least protective training loop. So the lesson is not "add more architecture." It
is the opposite: I want a learner that needs no delicate optimization at all — no learning-rate schedule
to thread, no initialization to survive, no early-stop gamble — and that is robust on noisy tabular data
out of the box.

Let me lay out the honest fork here rather than jump. Given "the failure was optimization, not bias," the
tempting move is to *keep* the transformer and fix its training: the editable `CustomModel` does let me
write a manual warmup-then-decay schedule inside the loop — a few hundred steps ramping the rate from near
zero, which is precisely the missing piece a from-scratch transduction model would have. I could size it:
with on the order of two hundred fifty steps per epoch, a warmup of, say, four thousand steps is about
sixteen epochs of ramp before the rate even reaches `1e-4`. But that is the move I should *not* make, and
the reason is in the numbers, not in taste. The transformer's problem is not a single missing schedule; it
is a *stack* of fragilities — initialization sensitivity, a 94%-FFN parameter mass with dropout switched
off, patience-5 cutting the run short, and a faint signal that gives the optimizer almost no gradient to
lean on. Patching in warmup addresses one of those while leaving me on the same high-wire, and worse, a
sixteen-epoch warmup fights directly against a patience-5 early stop that can terminate the run before the
schedule has even finished ramping. The other tempting patch — just lower the rate further or widen the
patience — is guessing hyperparameters on a model whose whole character is that it is hard to train. The
directed move is to change the *class* of learner so the entire family of optimization fragilities
disappears at once, not to bargain with it. That points away from sequence models entirely, back to the
consensus high-accuracy learner on structured data: a gradient-boosted decision tree.

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

Within the tree family there is still a fork worth resolving on paper: a random forest would also be
robust and initialization-free, so why boosting rather than bagging? A forest grows many *independent*
deep trees on bootstrap samples and averages them; its variance reduction comes from decorrelating
high-variance, low-bias trees. Boosting instead grows *dependent* shallow-ish trees, each fitting what
the ensemble still gets wrong, with a shrinkage rate that lets me take the additive fit arbitrarily
slowly. The decisive difference for this data is the control surface. A forest has no learning rate — each
tree contributes a full `1/T` share and I cannot dial the aggressiveness of the fit down toward a faint
signal; my only knobs are tree count and depth. Boosting hands me `learning_rate` and a per-tree early
stop, so I can make the ensemble crawl toward the signal and halt the instant validation stops improving
— exactly the "act only on strong, repeated signal" behavior a low-IC problem demands. And the forest's
own strength, deep high-variance trees, is a liability here: on a signal this faint, deep independent
trees mostly fit noise that averaging only partly cancels, whereas heavily shrunk boosting never lets any
single tree commit hard to noise in the first place. So boosting, not bagging.

I can even sanity-check the residual-fitting picture against the label pipeline to be sure I am not
fooling myself. The label passes through `CSRankNorm`, so per cross-section the targets are rank-
normalized to mean zero. The ensemble's constant initial prediction is therefore near zero, and the first
tree fits `y − 0 = y`. Take the degenerate one-leaf case with the penalties off: the optimal leaf value is
`w* = −G/H = −Σ(0 − y_i)/n = mean(y) ≈ 0`, which is exactly the right answer for a zero-mean target — the
model correctly does nothing until a split isolates a subpopulation whose mean return departs from zero.
That is the mechanism I want: it only acts where there is coherent signal, and sits still on noise.

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
`O(#data × #feature)`. On this training set that exact scan is punishing — with the same `~5·10⁵`
stock-days I estimated for the neural rung and 360 features, a single node's exact split search is on the
order of `1.9·10⁸` operations. The histogram engine cuts this to size: bucket each continuous feature into
a small fixed number of bins (255, so a bin index fits in a byte), and for a node make one pass
accumulating per-bin sum-of-gradients and count. Searching splits is then `O(#bin × #feature)` —
`255·360 ≈ 9.2·10⁴`, some three orders of magnitude below the exact scan, a rounding error next to the
build. One detail of the binning matters given that I am about to feed the trees *raw*, un-normalized features:
the bin edges are chosen once, up front, from the feature's empirical distribution — quantile-style, so
each of the 255 bins holds roughly equal data mass rather than equal width. That is exactly why I do not
need to standardize inputs for the tree: a monotone rescaling of a feature leaves its quantiles, and
therefore its bin assignments and every candidate split, unchanged. The histogram makes the model
scale-invariant by construction, which is the mechanical reason the `RobustZScoreNorm` the neural rung
depended on is not merely unnecessary here but, in its clipping, actively harmful (a point I come back to
at the processor edit). A parent node's histogram is the sum of its two children's, so I build only the
*smaller* child's histogram and recover the sibling by subtraction in `O(#bin)`. On top of that the engine grows leaf-wise
— best-first, splitting the single leaf with the largest loss reduction at each step — which reaches lower
training loss than level-wise at a fixed leaf budget, capped by `num_leaves` and `max_depth` to bound
overfit. These are the knobs the deployment exposes, and they are what make a thousand-tree ensemble
tractable on this data.

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
expressive trees — though note `2⁸ = 256 > 210`, so `max_depth` is the binding constraint on the deepest
paths while `num_leaves` caps the total, a deliberate pairing that lets the tree be unbalanced (deep
where signal warrants) without exploding. The two L-penalties are enormous — `lambda_l1 = 205.7`,
`lambda_l2 = 580.98` — and that is deliberate and characteristic of financial data, so let me work out
what they actually do to a leaf. The regularized leaf value is
`w* = −sign(G)·max(|G| − λ1, 0) / (H + λ2)`. The L1 term is a hard gate: a leaf's gradient sum `G` must
exceed `205.7` in magnitude before the leaf gets *any* nonzero value at all. With rank-normalized targets
whose per-sample gradient is `O(1)`, that means a leaf needs on the order of a couple hundred samples all
pulling their residual the same direction before it is allowed to move the prediction — pure noise, whose
signs cancel, never clears the bar. The L2 term then shrinks whatever survives: for a leaf of, say, a
thousand samples, `H ≈ 1000` (unit Hessian), so the denominator becomes `1000 + 581 = 1581` and the leaf
value is scaled to `1000/1581 ≈ 0.63` of its unpenalized size — a further 37% haircut on top of the L1
gate and the `0.0421` shrinkage. Stack those three and each tree moves the ensemble a genuinely tiny,
noise-averaged step. That is the whole design: the trees only act on the strongest, most repeated signal
and ignore the vast noise floor. It is worth counting the nominal capacity so I appreciate how much
taming these penalties are doing: up to a thousand trees of up to two hundred ten leaves each is on the
order of `2·10⁵` leaf values, grown *adaptively* to the data rather than fixed in advance — normally a
recipe for overfit. The reason it is not is that essentially none of those leaf values are free. The L1
gate zeros most candidate leaves outright, the L2 shrinkage and the `0.0421` rate scale down the
survivors, early stopping caps the round count well below a thousand, and each surviving leaf value is an
average over hundreds of samples rather than a fitted free weight. The effective degrees of freedom are a
tiny fraction of the nominal count — precisely the property I want against a variance-limited target, and
precisely what the brittle transformer, with its 94%-FFN mass and dropout switched off, did *not* have.
`colsample_bytree = 0.8879` and `subsample = 0.8789` (with the default
bagging) add column- and row-level stochasticity so successive trees decorrelate and the ensemble
generalizes. `objective = mse` matches the regression label. Every one of these is pushed toward *more*
regularization than a default tabular task would use, because the IC ceiling here is low and the failure
mode is variance, not bias.

There is one more task-specific edit that matters, and it is exactly the kind of same-named-but-different
detail I have to get right: this rung does *not* leave the workflow processors alone. The default
workflow applies a neural-model preprocessing block — `RobustZScoreNorm` (clip-and-standardize the
features) and `Fillna` as `infer_processors`. Those are there for the gradient-descent models, which need
standardized inputs to train stably. A tree does not: it splits on order statistics, so monotone
per-feature rescaling is irrelevant to it. But `RobustZScoreNorm` is not purely monotone — it *clips* to
roughly `±3` robust standard deviations before standardizing, and clipping is exactly the non-monotone
operation that hurts a tree: every value beyond the clip boundary is flattened to the same number, so the
tree can no longer order the extremes or split *among* them. In return prediction the extreme moves are
often the informative ones, so that clipped tail is signal the tree is being denied. So the lgbm edit
sets `infer_processors: []` — feed the trees the raw features — and keeps only the label-side
`DropnaLabel` + `CSRankNorm`. Dropping the block also drops `Fillna`, and I should check that is safe:
without it the raw feature matrix still carries NaNs, but the histogram engine handles missing values
natively — it holds them out of the binned histogram and, at each split, learns which child the NaNs
should default to by trying both and keeping whichever gives more gain. So the tree treats "this ratio is
missing" as its own routable signal rather than needing it imputed to a fake number, which is strictly
better than `Fillna`'s zero-fill would have been. The neural model, by contrast, cannot feed a NaN through
a matrix multiply at all, which is why `Fillna` is non-negotiable *for it* and discardable here. This is a genuine deviation from the transformer rung, which used the
default processor block unchanged, and it is the correct one: the preprocessing that *helped* the neural
model (bounded, standardized inputs for stable gradients) would have quietly *hurt* the tree (order
destroyed at the tails). The full scaffold module and the processor edit are in the answer.

So the delta from the Transformer rung is total: I throw out the sequence model and its delicate Adam
training, treat the 60×6 window as a flat tabular row, and fit a heavily-regularized histogram GBDT that
needs no warmup, no initialization luck, and no early-stop gamble — while stripping the
neural-only feature normalization so the trees see the raw signal.

What do I expect against the Transformer's measured numbers, and what is the falsifiable claim? First and
most important, the IR must go *positive* on csi300 — the transformer's −0.44 was a near-noise ranking
paying churn costs, and a GBDT that actually forms signal should rank well enough to earn back the costs;
if the csi300 IR comes back negative, the tree failed to find signal and my robustness argument is wrong.
Second, the IC should climb substantially from the transformer's 0.0117 toward the low-single-digit-percent
range a working model on this data reaches — anything still near 0.01 would mean the loss of the temporal
bias cost more than the robustness gained. Third, where I am genuinely uncertain: on csi100, even the transformer's IC
(0.020) outran its csi300 IC (0.012), yet its return was the *most* negative of the three; csi100 is a
smaller, harder universe where the IC-to-IR decoupling I noticed above says ranking quality does not
translate into portfolio return the way it does on csi300, because fifty names out of a hundred is a much
larger slice of the universe and the TopkDropout churn bites harder. So my honest expectation is: a clear
positive csi300 IR and a real IC jump (the robustness fix working), but I would not be surprised if
csi100's portfolio return stays weak — even negative — even as its IC improves. And if so, the diagnosis
for the next rung is that the tree's per-row tabular view, however robust, still leaves ranking quality on
the table that a temporal model trained *robustly* (not the brittle transformer) could recover. That is
the fork the next rung takes.
