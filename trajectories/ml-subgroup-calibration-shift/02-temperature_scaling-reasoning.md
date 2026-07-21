The subgroup-aware run told me what I half-feared, in the one column I am graded on. The worst-group ECE
came in at 0.4938 on Adult, 0.4972 on COMPAS, 0.3138 on Law School — mean 0.4349 — while the Brier
scores are respectable (0.433 / 0.360 / 0.317) and `subgroup_auroc` is untouched at 0.8590 / 0.8819 /
0.7486, exactly as the monotone-map argument promised. So the per-group temperatures did something
sensible on average but the *worst* subgroup barely moved, and on COMPAS it is essentially as bad as the
raw classifier would be.

The feedback carries a second number per dataset that decomposes the first. `max_subgroup_gap` is the
spread between the worst and best subgroup ECE, so `best-subgroup ECE = worst_group_ece −
max_subgroup_gap`. On Adult, `0.4938 − 0.4361 = 0.058`: one subgroup essentially calibrated while the
worst sits at 0.494, an enormous internal spread. On COMPAS, `0.4972 − 0.1417 = 0.356`: a tight band in
which everyone is badly calibrated. On Law School, `0.3138 − 0.1947 = 0.119`, intermediate. That says
the three datasets fail for two different reasons. Adult's problem is *heterogeneity* — nearly honest on
one cell, wildly off on another, precisely what a per-group correction was invented for — and yet my
method still posted 0.494 on the worst cell, so the machinery did not reach the group that needed it.
COMPAS's problem is *uniform badness*: the whole population is mis-scaled, with little group-to-group
structure for any per-group knob to exploit. Law School sits between.

The diagnosis of my own method follows. The worst subgroup is the small one, and `α_g = n_g/(n_g + 200)`
pulls it hard toward the global temperature: a 30-to-50-point cell keeps only `0.13`–`0.20` of its own
local fit, so four-fifths of its temperature *is* the global temperature already. On the group that
decides my metric I am effectively doing global temperature scaling with a thin local perturbation — the
shrinkage that protects me from a noisy local fit also denies the worst group any real individual
correction. That explains Adult (the exploitable cell was small, so shrinkage routed it through the
global scalar) and COMPAS differently (no group structure to exploit, and the global temperature inside
the shrunk fit cannot fix a scale error afflicting every cell on the shifted tail). Either way the
per-group degrees of freedom paid for themselves in Brier and bought almost nothing in worst-group ECE.

Read against the Murphy decomposition, the Brier column says something about whether *any* post-hoc
scalar can rescue COMPAS. Brier splits as reliability minus resolution plus uncertainty, and the
uncertainty term `π(1−π)` for the test base rate is at most `0.25`. COMPAS's Brier of 0.360 sits above
that ceiling by about a tenth, so reliability minus resolution is positive by at least that much: the
miscalibration penalty is outrunning the discrimination the ranking earns, even though `subgroup_auroc`
says the ordering is a healthy 0.88. Adult is worse, 0.433 against the same ceiling. A Brier above the
uncertainty floor is the fingerprint of a *transfer* failure — the temperature that minimized
calibration-region NLL is the wrong temperature for the tail, and no per-group cleverness on the
calibration split changes the tail it is applied to.

So before I climb to a richer map I need a fact I do not have: how much of the 0.4349 is the subgroup
machinery and how much is just what a single global temperature does here? The group method never runs
as a clean global scalar — it always layers per-group fits on top, at roughly four-fifths global plus
one-fifth local noise on the worst cell — so I cannot tell whether that one-fifth helps, hurts, or does
nothing. Pushing the shrinkage harder (raise `k`) is just an asymptotic approach to the global scalar
dressed up, and tuning `k` on a shifted split with no honest cross-validation is the knob the design has
avoided; estimating `k` from the between-group variance I already rejected with only a handful of
groups; going richer per-group adds variance on exactly the small worst cell. The clean isolation is to
strip every per-group parameter and fit one temperature for everyone — simpler than what I ran, the
lowest-variance map I can fit here, which is the property that should transfer under shift.

That floor is the global temperature already running inside the group method, with the group loop
removed: `q = σ(z/T)`, `z = logit(p)`, one positive scalar fit by minimizing the calibration-split NLL,
monotone in `z` so ranking, accuracy, and `subgroup_auroc` are exactly preserved, `groups` accepted and
ignored. Its value is the variance. A single `T` is fit on all the calibration points at once; at these
logit magnitudes the per-example Fisher information is about `0.36`, so a pooled fit on a few thousand
points has curvature in the thousands and `Var(log T) ≈ 1/3000 ≈ 3·10⁻⁴`, a standard deviation near
`0.03` — the global temperature pinned to about `±3%`, roughly a tenth the estimation noise of the worst
subgroup's `±35%` per-group fit. So whatever error the control posts on the worst group is almost pure
bias-plus-shift with the estimation-variance term all but zeroed out — the clean decomposition I need.
It does not promise the control will be *good*: the pooled `T` still minimizes calibration-region NLL
and the shifted tail is a different distribution, so a bias remains that I cannot see from the
calibration split. Low estimation variance is necessary for transfer, not sufficient — but it means that
if the worst-group ECE stays high, I can pin the blame on shape-and-shift rather than noisy fitting.

Each dataset is fit and evaluated on its own at seed 42, so there are three independent single-scalar
fits and the mean is their average — and since the datasets fail for different reasons, I should expect
three separate diagnoses rather than one verdict.

I should be clear-eyed about the ceiling this control will hit, because that ceiling is the real reason
to run it. A single `T` applies *one* correction to the entire score range. Suppose the mid-band scores
are about right but the extremes are blown out: `p = 0.7 → σ(0.847/2) = 0.604` and
`p = 0.99 → σ(4.595/2) = 0.909` under `T = 2`. The *ratio* of those two corrections is fixed by the one
number `T` — I cannot leave `0.7` roughly alone while pulling `0.99` down hard, because any `T` large
enough to pull the extreme over-softens the middle, and any `T` gentle on the middle barely touches the
extreme. One slope cannot fit a distortion that is steep in one region and flat in another, or
asymmetric between the ends. If the worst subgroups carry that kind of non-uniform distortion, the
global scalar bottoms out well short of calibration no matter how perfectly I fit `T`.

Implementation is the bare version of what ran globally inside the group method: clip `p` into
`[ε, 1−ε]`, take `z = logit(p)`, minimize the NLL of `σ(z/T)` over `log T ∈ [−3, 3]` by a bounded 1-D
search (fit in `log T` because `T > 0` is a positivity constraint and `T` is multiplicative; bound the
box so `T` cannot run away on a flat objective), and at predict time divide the logit by the fitted `T`,
apply `σ`, clip back. The full module is in the answer.

What do I expect against the numbers I just measured? Because the group method already routes its worst
(small) subgroup through roughly four-fifths global temperature, plain global scaling should post a
worst-group ECE *close to* the group method's — not dramatically better, but plausibly a touch better if
the leftover local weight was mildly hurting. There is a sharper reason to lean toward the control
winning: worst-group ECE is a *maximum* over cells. Even if the per-group perturbations were mean-zero —
just jitter — injecting independent noise into each cell's error can only *raise* the expected maximum,
because the max latches onto whichever cell the noise pushed highest. So even in the charitable world
where the per-group machinery is unbiased, it is expected to lose on this metric by construction, and
the global scalar removes that inflation by refusing to jitter the cells. By the same token stripping
the per-group fits should tighten `max_subgroup_gap`, Brier should move only slightly (a single `T`
still minimizes pooled NLL), and `subgroup_auroc` must be *identical* — both maps monotone, AUROC
invariant to monotone transforms — so if that column moves I have a bug.

Whichever way the worst-group ECE falls, it tells me the same thing about where to go. If even the
lowest-variance global scalar cannot get Adult and COMPAS much below ~0.45, the bottleneck is not
parameter count or the per-group split — it is the *shape* of the global map, which can divide the logit
by a constant but cannot bend, cannot gather extreme scores back toward the middle, cannot fit an
asymmetric distortion. The shape freedom is likelier to help on Adult, whose heterogeneity implies at
least one non-uniform cell, than on COMPAS, whose uniform badness may resist any monotone reshaping. So
the next move is to keep the low-variance, group-agnostic posture but give the global map a richer,
still-monotone shape.
