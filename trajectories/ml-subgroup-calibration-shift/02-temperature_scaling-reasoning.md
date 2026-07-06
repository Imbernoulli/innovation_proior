The subgroup-aware run told me what I half-feared, and it told me in the one column I am graded on. The
worst-group ECE came in at 0.4938 on Adult, 0.4972 on COMPAS, 0.3138 on Law School — mean 0.4349 — and
the tell is not any single number, it is the *shape*. The Brier scores are respectable (0.433 / 0.360 /
0.317, mean 0.370) and the `subgroup_auroc` is untouched at 0.8590 / 0.8819 / 0.7486 exactly as the
monotone-map argument promised. So the per-group temperatures did something sensible on average — Brier
is a proper score and it is fine — but the *worst* subgroup barely moved, and on COMPAS the worst-group
ECE is essentially as bad as the raw classifier would be.

Before I decide what to do about that, let me get more out of the feedback than the headline column,
because there is a second measured number per dataset that decomposes the first. The `max_subgroup_gap`
is defined here as the spread between the worst and best subgroup ECE, so `best-subgroup ECE =
worst_group_ece − max_subgroup_gap`, and I can just subtract. On Adult, `0.4938 − 0.4361 = 0.058`: one
subgroup is essentially calibrated at an ECE of six hundredths while the worst sits at 0.494 — an
enormous internal spread. On COMPAS, `0.4972 − 0.1417 = 0.356`: the best subgroup is at 0.356 and the
worst at 0.497, a *tight* band in which everyone is badly calibrated. On Law School, `0.3138 − 0.1947 =
0.119`: best 0.119, worst 0.314, intermediate. That decomposition is worth more than the mean, because
it says the three datasets fail for two different reasons. Adult's problem is *heterogeneity*: the model
is nearly honest on one cell and wildly off on another, which is precisely the situation a per-group
correction was invented for — and yet my per-group method still posted 0.494 on the worst Adult cell,
so the machinery did not reach the group that needed it. COMPAS's problem is *uniform badness*: the
whole population, best cell included, is mis-scaled, and there is very little group-to-group structure
for any per-group knob to exploit; the shifted COMPAS tail is simply hard for everyone. Law School sits
between the two.

Now the diagnosis of my own method sharpens. The worst subgroup is the small one, and the empirical-
Bayes weight `α_g = n_g/(n_g + 200)` pulls the small one hard toward the global temperature: a 30-to-50-
point cell keeps only `α = 0.13`–`0.20` of its own local fit, so roughly four-fifths of its temperature
*is* the global temperature already. On the group that decides my metric I am therefore effectively
doing global temperature scaling with a thin local perturbation on top — the shrinkage that protects me
from a noisy local fit also denies the worst group any real individual correction. That explains Adult:
the heterogeneity was there to exploit, but the exploitable cell was small, so shrinkage routed it
through the global scalar and left 0.494. It explains COMPAS differently: there was no group structure
to exploit in the first place, so even a perfect per-group scheme could not have helped, and the global
temperature riding inside the shrunk fit could not fix a scale error that afflicts every cell on the
shifted tail. Either way, the per-group degrees of freedom paid for themselves in Brier and bought me
almost nothing in worst-group ECE.

There is one more thing the Brier column tells me if I read it against the Murphy decomposition, and it
bears on whether *any* post-hoc scalar can rescue COMPAS. Brier splits as reliability minus resolution
plus uncertainty, and the uncertainty term is `π(1−π)` for the test base rate `π`, which is at most
`0.25`. COMPAS's Brier of 0.360 sits *above* that ceiling by about a tenth, so reliability minus
resolution must be positive by at least that much: the miscalibration penalty is outrunning the
discrimination the ranking earns, even though `subgroup_auroc` says the ordering is a healthy 0.88.
Adult is worse still, a Brier of 0.433 against the same 0.25 ceiling. A Brier above the uncertainty floor
is the fingerprint of a map that is shedding calibration faster than the ranking pays it back — which on
the shifted tail is not a fitting failure but a *transfer* failure: the temperature that minimized
calibration-region NLL is simply the wrong temperature for the tail, and no amount of per-group
cleverness on the calibration split changes the tail it is applied to. That reframes what the control is
testing. It is not asking whether one number can beat many numbers on the calibration set; it is asking
how much of the residual half-point of worst-group ECE is irreducible shift that even the
best-transferring map must eat, versus shape that a richer map could still remove.

So before I climb to a richer map I need a baseline fact I do not actually have yet: *how much of that
0.4349 is the subgroup machinery, and how much is just what a single global temperature does here?* The
group method degenerates to global temperature scaling when the groups carry no usable local signal, but
it never runs that way cleanly — it always layers the per-group fits on top, and on the worst cell it is
running at roughly four-fifths global plus one-fifth local noise. I cannot tell from the numbers I have
whether that leftover one-fifth is helping the worst group, hurting it, or doing nothing, because I have
never measured the clean global control. Let me weigh the alternatives to running that control, because
"step back" should be a decision, not a retreat. One option is to push the shrinkage harder — raise `k`
from 200 to, say, 1000 — so the small cells pool almost entirely to global. But that is just an
asymptotic approach to the global scalar dressed up as a per-group method, and worse, it introduces a
hyperparameter I would be tuning on a shifted split with no honest cross-validation, exactly the kind of
knob the whole design has been avoiding. A second option is to estimate `k` from the between-group
variance and let the data set the shrinkage — but I already rejected that on the previous rung, because
with only a handful of groups the between-group variance is itself a high-variance estimate, the very
disease I am treating. A third option is to go *richer* per-group, more parameters per cell — but the
worst cell is the small cell, and more parameters there is more of the variance that just cost me the
metric. The cleanest way to isolate the control is neither to tune nor to enrich: it is to strip every
per-group parameter and fit one temperature for everyone. It is simpler than what I just ran, and that
is the point — it is the floor against which the subgroup machinery either justifies itself or does not,
and it is the lowest-variance object on the whole ladder, which is exactly the property that should
transfer under shift.

Let me re-derive that floor from scratch, because I want it airtight. The raw `p` is over-confident in
the standard way: a log-loss-trained classifier, once it is classifying almost everything correctly, can
still lower its loss by pushing probabilities toward 0 and 1, overfitting NLL long after 0/1 error has
flattened, and the excess goes into confidence. Crucially the failure is *scale*, not order — the
ranking is intact, which `subgroup_auroc` keeps confirming at 0.86/0.88/0.75 across every method I run.
So I want the smallest knob that fixes a scale and leaves the order alone. Scale does not live in `p`,
which is squashed into `[0,1]`; it lives in the logit `z = logit(p) = log(p/(1−p))`, where `σ` and
`logit` are the monotone bijection between `[0,1]` and the real line. The minimal scale correction is to
divide the logit by a positive number: `q = σ(z/T)`. `T = 1` is the identity floor; `T > 1` shrinks
every logit toward zero and pulls every `q` toward `1/2`, softening over-confidence and raising entropy;
`T < 1` sharpens; `T → ∞` collapses to `1/2`, `T → 0` snaps to hard 0/1. And the property that makes
this safe on a benchmark whose `subgroup_auroc` I must not damage: `z/T` is monotone increasing in `z`
for any `T > 0`, so it never reorders examples and never moves the `z = 0 ↔ p = 0.5` boundary — the
predicted class, accuracy, and AUROC are exactly preserved, only the confidences are softened. This is
temperature scaling, the one-parameter special case of Platt's `σ(a·z + b)` with `a = 1/T` and the
intercept `b` dropped. Dropping `b` is deliberate: a nonzero intercept moves the boundary `a·z + b = 0`
off `z = 0` and would let the recalibration change predictions, and an extra parameter is extra variance
I refuse to pay when the whole lesson of the last run is that variance under shift is what hurts.

Is dividing the logit by a scalar the *right* correction or just a convenient one? Pin down what it
optimizes. I fit `T` by minimizing the calibration-split binary NLL `−mean(y log q + (1−y) log(1−q))`,
because NLL is a proper scoring rule — in expectation minimized exactly when the reported probability is
the true conditional — so descending it pushes `q` toward calibration. ECE bins and is non-
differentiable, so I fit NLL and *measure* ECE, never the other way around. And the family "scale the
logits" is not arbitrary: among all per-example distributions that are valid probabilities and that
match one moment — the average true-class logit equals the average expected logit under `q` — the
maximum-entropy one is the softmax of `λ z`. Set up the Lagrangian `L = −Σ_i Σ_k q_i^(k) log q_i^(k) + λ
Σ_i [Σ_k z_i^(k) q_i^(k) − z_i^(y_i)] + Σ_i β_i (Σ_k q_i^(k) − 1)`; stationarity gives `−log q_i^(k) − 1
+ λ z_i^(k) + β_i = 0`, so `q_i^(k) = exp(λ z_i^(k) + β_i − 1)`, and normalizing kills `β_i` to leave
`q_i^(k) = softmax(λ z_i)^(k)`; write `λ = 1/T` and that is logits-over-`T`. In binary form, `q = σ(z/T)`.
So the single scalar is the lone Lagrange multiplier of the lone moment constraint — the honest minimal
fix for exactly the scale error I diagnosed. There is essentially no capacity here to overfit the
calibration split, which is precisely why I expect it to *survive the shift* better than the per-group
fits did: a single number estimated from the whole calibration set has the lowest variance of anything
on this ladder, and low variance is what transfers when the test tail is shifted.

Let me put a number on "lowest variance," because it is the whole reason to expect the control to
transfer. A single `T` is fit on all the calibration points at once; reusing the per-example Fisher
information I estimated on the last rung — about `0.36` per point at these logit magnitudes — a pooled
fit on a few thousand points has curvature in the thousands, so `Var(log T)` is on the order of
`1/3000 ≈ 3·10⁻⁴`, a standard deviation near `0.03` — the global temperature is pinned to about `±3%`.
Set that beside the worst subgroup's per-group temperature, which I estimated at `±35%` off its thirty
to fifty points: the global scalar carries roughly a tenth the estimation noise on the very cell that
decides the metric. So whatever error the control posts on the worst group is almost pure bias-plus-
shift, with the estimation-variance term all but zeroed out — which is exactly the clean decomposition I
need. It does *not* promise the control will be good: the pooled `T` still minimizes calibration-region
NLL, and the shifted tail is a different distribution, so there remains a bias I cannot see from the
calibration split. Low estimation variance is necessary for transfer, not sufficient. But it means that
if the worst-group ECE stays high, I can pin the blame on shape-and-shift rather than on noisy fitting —
and that attribution is the thing the last run could not give me, because it never ran a clean scalar.

One clarification about what "the control" even is, since it changes how I read the coming numbers. Each
dataset is fit and evaluated on its own at seed 42, so there is not one temperature but three
independent single-scalar fits, and the table's mean is just their average. That matters because the
best-subgroup decomposition already told me the three datasets fail for different reasons — Adult by
heterogeneity, COMPAS by uniform badness, Law School in between — so I should not expect a single verdict
from the control. The honest question is per-dataset: on each one, does one number reach the worst cell,
and if not, is the residual the kind a *bend* could remove or the kind the shift makes irreducible? That
turns three numbers into three separate diagnoses, which is far more than "did it beat 0.4349."

I should also be clear-eyed, though, about the ceiling this control is going to hit, because that ceiling
is the real reason I am running it — not to win, but to expose where the win has to come from. A single
`T` can only apply *one* correction to the entire score range. Watch what that costs when the true
distortion is not uniform. Suppose on some dataset the scores in the middle band are already about right
but the scores near the extremes are the ones blown out. Push two of them through `T = 2`: `p = 0.7 →
σ(0.847/2) = 0.604`, and `p = 0.99 → σ(4.595/2) = 0.909`. The single temperature moved the mid score by
about a tenth and the extreme score by about eight hundredths, but the *ratio* of those corrections is
fixed by the one number `T` — I cannot ask it to leave `0.7` roughly alone while pulling `0.99` down to,
say, `0.82`, because any `T` large enough to pull the extreme hard also over-softens the middle, and any
`T` gentle on the middle barely touches the extreme. One slope cannot fit a distortion that is steep in
one region and flat in another, or asymmetric between the low and high ends. That is the structural
poverty of a scalar, and if the worst subgroups on Adult and COMPAS carry exactly that kind of
non-uniform distortion, the global scalar will bottom out well short of calibration no matter how
perfectly I fit `T`.

Implementation is the bare version of what was already running globally inside the group method, with the
group loop removed. Clip `p` into `[ε, 1−ε]` with `ε = 1e-6` so the logit is finite; take `z = logit(p)`;
minimize the NLL of `σ(z/T)` over `log T ∈ [−3, 3]` (i.e. `T ≈ [0.05, 20]`) by a bounded 1-D scalar
search — fit in `log T` because `T > 0` is a positivity constraint I would rather not babysit and `T`
lives on a multiplicative scale where `log T` is the natural unconstrained coordinate, and bound the box
so the search stays well conditioned and `T` cannot run away on a flat objective. At predict time take
the logit of the (clipped) input, divide by the fitted `T`, apply `σ`, clip the output back into
`[ε, 1−ε]`. `groups` is accepted and ignored — this is a group-agnostic method by construction, which is
exactly the control I want. The full scaffold module is in the answer.

Now the falsifiable part, stated against the numbers I just measured. The first prediction is the
sharpest, and it follows directly from the `α` arithmetic: because the group method already routes its
*worst* (small) subgroup through roughly four-fifths global temperature, plain global temperature
scaling should post a worst-group ECE that is *close to* the group method's — not dramatically better,
because it does not add anything the worst group was not already getting, but plausibly a touch better
on Adult and COMPAS if the leftover one-fifth local weight on the mid-sized groups was mildly hurting
the worst group rather than helping it. Concretely I expect Adult and COMPAS worst-group ECE to land
near or just below the 0.494 / 0.497 the group method posted, and Law School near its 0.314. If instead
global temperature scaling comes in clearly *better* than the group method on worst-group ECE, that is
the strong confirmation that the per-group machinery was actively harmful under shift — that the
shrinkage did not shrink hard enough and the small-group noise leaked into the metric.

There is a subtler reason to lean toward the control winning this comparison, and it comes from the
*shape* of the objective rather than from any bias. Worst-group ECE is a *maximum* over cells. Suppose,
generously, that the per-group perturbations are mean-zero — that the local fits neither systematically
help nor hurt, just jitter each cell's ECE up or down by the estimation noise I quantified. A maximum
over jittered cells is not itself mean-zero: injecting independent noise into each cell's error can only
*raise* the expected maximum, because the max latches onto whichever cell the noise pushed highest. So
even in the charitable world where the per-group machinery is unbiased, it is expected to lose on this
particular metric by construction, purely because the metric reads the worst cell and per-group noise
inflates the worst cell. The global scalar removes that inflation by refusing to jitter the cells at all.
That is a cleaner prediction than "the local fits are biased": the group method should trail the control
on worst-group ECE even if its per-group temperatures are, on average, right. The second
prediction concerns `max_subgroup_gap`: stripping the per-group fits should, if anything, *tighten* the
spread, because the noisy local perturbations on the mid and small cells were adding cell-to-cell
scatter; so I expect the gap to come in at or below the group method's 0.436 / 0.142 / 0.195. The third
prediction is that Brier should move only slightly and possibly a hair *worse*: a single global `T`
minimizes pooled NLL, so its Brier should be in the same neighborhood as the group method's 0.370 mean,
maybe marginally up on the datasets where per-group fitting genuinely helped the population-weighted
average. The fourth, which I am most confident in: `subgroup_auroc` must be *identical* to the group
method's 0.8590 / 0.8819 / 0.7486, because both maps are monotone in the logit and AUROC is invariant to
any monotone transform — if that column moves at all, I have a bug.

The reason this control matters for the next rung is that whichever way the worst-group ECE falls, it
tells me the same thing about *where* to go. If even a single, lowest-variance global scalar cannot get
worst-group ECE much below ~0.45 on Adult and COMPAS, then the bottleneck is not the number of
parameters or the per-group split — it is the *shape* of the global map. Temperature scaling can only
divide the logit by a constant; it cannot bend, cannot gather extreme scores back toward the middle,
cannot fit an asymmetric distortion, as the two-point illustration just showed. And the best-subgroup
decomposition tells me the shape freedom is likelier to help on Adult, where the heterogeneity implies
at least one cell whose distortion is genuinely non-uniform, than on COMPAS, where the uniform badness
may resist any monotone reshaping. The moment the global scalar bottoms out, the next move is to keep
the low-variance, group-agnostic posture that just paid off under shift but give the *global* map a
richer, still-monotone shape — which is the axis I will climb next.
