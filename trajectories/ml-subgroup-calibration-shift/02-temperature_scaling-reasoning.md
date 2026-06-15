The subgroup-aware run told me what I half-feared, and it told me in the one column I am graded on. The
worst-group ECE came in at 0.4938 on Adult, 0.4972 on COMPAS, 0.3138 on Law School — mean 0.4349 — and
the tell is not any single number, it is the *shape*. The Brier scores are respectable (0.433 / 0.360 /
0.317, mean 0.370) and the `subgroup_auroc` is untouched at 0.8590 / 0.8819 / 0.7486 exactly as the
monotone-map argument promised. So the per-group temperatures did something sensible on average — Brier
is a proper score and it is fine — but the *worst* subgroup barely moved, and on COMPAS the worst-group
ECE is essentially as bad as the raw classifier would be. That is precisely the structural failure I
flagged before running it: the worst subgroup is the small one, the small one is exactly the group the
empirical-Bayes weight `α_g = n_g/(n_g + 200)` pulls hard toward the global temperature, so on the group
that decides my metric I am effectively *already doing global temperature scaling* — the shrinkage that
protects me from a noisy local fit also denies the worst group any individual correction. The per-group
degrees of freedom paid for themselves in Brier and bought me almost nothing in worst-group ECE under
this shift. And the COMPAS number worries me more specifically: 0.497 worst-group ECE with a `brier` of
0.360 means the model is well-ordered but badly mis-scaled on one subgroup, and even the global
temperature riding inside the shrunk per-group fit did not fix the scale on the shifted COMPAS tail.

So before I climb to a richer map I need to know a baseline fact I do not actually have yet: *how much of
that 0.4349 is the subgroup machinery, and how much is just what a single global temperature does here?*
The group method degenerates to global temperature scaling when the groups carry no usable local signal,
but it never runs that way cleanly — it always layers the per-group fits on top, and the Adult/COMPAS
worst-group numbers (0.494 / 0.497) are slightly *worse* than I would expect a clean global scalar to
post, which makes me suspect the leftover local weight on the mid-sized groups is mildly hurting the
worst group rather than helping it. I cannot diagnose the next step honestly until I have isolated the
plain global scalar as a control. So this rung deliberately steps *back*: strip out every per-group
parameter and fit one temperature for everyone. It is simpler than what I just ran, and that is the
point — it is the floor against which the subgroup machinery either justifies itself or does not.

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
sharpest: because the group method already routes its *worst* (small) subgroup through essentially the
global temperature, plain global temperature scaling should post a worst-group ECE that is *close to* the
group method's — not dramatically better, because it does not add anything the worst group was not
already getting, but plausibly a touch better on Adult and COMPAS if the leftover local weight on the
mid-sized groups was mildly hurting the worst group. Concretely I expect Adult and COMPAS worst-group
ECE to land near or just below the 0.494 / 0.497 the group method posted, and Law School near its 0.314.
If instead global temperature scaling comes in clearly *better* than the group method on worst-group ECE,
that is the strong confirmation that the per-group machinery was actively harmful under shift — that the
shrinkage did not shrink hard enough and the small-group noise leaked into the metric. The second
prediction is that Brier should move only slightly: a single global `T` minimizes pooled NLL, so its
Brier should be in the same neighborhood as the group method's 0.370 mean, maybe marginally worse on the
datasets where per-group fitting genuinely helped the average. The third, which I am most confident in:
`subgroup_auroc` must be *identical* to the group method's 0.8590 / 0.8819 / 0.7486, because both maps
are monotone in the logit and AUROC is invariant to any monotone transform — if that column moves at all,
I have a bug. The reason this control matters for the next rung is that whichever way the worst-group ECE
falls, it tells me the same thing about *where* to go: if even a single, lowest-variance global scalar
cannot get worst-group ECE much below ~0.45 on Adult and COMPAS, then the bottleneck is not the number
of parameters or the per-group split — it is the *shape* of the global map. Temperature scaling can only
divide the logit by a constant; it cannot bend, cannot gather extreme scores back toward the middle,
cannot fit an asymmetric distortion. The moment the global scalar bottoms out, the next move is to keep
the low-variance, group-agnostic posture that just paid off under shift but give the *global* map a
richer, still-monotone shape — which is the axis I will climb next.
