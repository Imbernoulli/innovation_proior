The research question is about subgroups, so I start by attacking subgroups head-on, and the floor I
build from is the identity fill: clip `p` into `[ε, 1−ε]` and hand it back unchanged. That floor is not
a real method, it is the absence of one — it reports the base classifier's raw probabilities and lets
the metric measure how wrong they are. The task ships a frozen, deliberately miscalibrated classifier,
so the raw `p` is over-confident and the identity will post a high worst-group ECE: the ECE bins
confidence into 15 equal-width buckets and reads the gap between mean confidence and empirical accuracy
in each, and an over-confident model dumps its mass into the top buckets, where reported confidence is
near `0.95` but the true positive rate sits well below, so those buckets carry a large gap and the worst
subgroup carries the largest. The whole point is to design the map that pulls those numbers back onto
the diagonal, *inside each subgroup*, since the worst subgroup is what I am graded on. My instinct is
that the strongest first move lets each subgroup get its own correction — but the subgroup machinery is
exactly where this can go wrong under the shift, so I want to see the failure mode coming before I run
anything.

First, why is `p` wrong, and which way? It is the standard story for a model trained to minimize
log-loss: once it is classifying almost everything correctly, the loss can still fall by shoving
probabilities toward 0 and 1, overfitting log-loss long after 0/1 error flattened, and the excess goes
into confidence. The useful part is *which* way — the dominant failure is that the overall *scale* of
the score is too big, the model uniformly too sure, not that the ranking is scrambled. If the ordering
were broken I would need something that can reorder examples; but the base pipeline is a logistic
regression that ranks fine, and the diagnostic `subgroup_auroc` confirms the ranking is intact and a
monotone calibrator cannot touch it. So I want the smallest knob that fixes a scale.

Scale does not live in `p` — squashed into `[0,1]`, a "scale" there is not well defined. It lives in the
logit `z = logit(p) = log(p/(1−p))`, where `σ` and `logit` are the monotone bijection between `[0,1]`
and the real line. So the cleanest scale knob is to divide the logit by a positive number, `q = σ(z/T)`.
Take `p ∈ {0.5, 0.9, 0.99}` with logits `z = 0, 2.197, 4.595` and `T = 2`: `q = σ(z/2) = 0.500, 0.750,
0.909`. The `0.5` example does not move (its logit is zero), the `0.9` softens to `0.75`, the `0.99` to
`0.909` — order preserved, the boundary at `p = 0.5` nailed, confidences pulled toward the middle by an
amount that grows with how extreme they were. `T = 1` is the identity floor I am replacing; `T > 1`
shrinks every logit toward zero so every `q` moves toward `1/2`, softening over-confidence and raising
entropy; `T < 1` sharpens. And `z/T` is monotone increasing in `z` for any `T > 0`, so it never reorders
examples and never moves the `z = 0 ↔ p = 0.5` boundary — the predicted class, accuracy, and
`subgroup_auroc` are untouched. One positive scalar with no risk to ranking. This is temperature
scaling, the one-parameter special case of Platt's `q = σ(a·z + b)` with `a = 1/T` and `b` dropped. I
drop `b` deliberately: a nonzero intercept moves the boundary `a·z + b = 0` off `z = 0` and would let
recalibration change predictions, and every extra parameter is variance I will have to pay for — which
matters enormously once I split by subgroup.

Is dividing the logit by a scalar the *right* correction or just a convenient one? I fit `T` by
minimizing the calibration-split NLL — binary cross-entropy — because NLL is a proper scoring rule,
minimized in expectation exactly when the reported probability is the true conditional, so descending it
pushes `q` toward calibration; the binned ECE is non-differentiable, so I fit NLL and *measure* ECE. And
"scale the logits" is not arbitrary: among all valid per-example distributions matching one moment — the
average true-class logit equals the average expected logit under `q` — the maximum-entropy one is the
softmax of `λ z` (the lone Lagrange multiplier of the lone moment constraint), and in binary form with
`λ = 1/T` that is exactly `q = σ(z/T)`. So a single temperature is the maximum-entropy distribution that
corrects nothing but the average logit scale, precisely the failure I diagnosed, and it carries
essentially no capacity to overfit the calibration split.

So a global temperature is well-founded, but it is one number for everyone, and my objective is the
*worst subgroup*. Different subgroups genuinely need different softening — their score distributions and
base rates differ, so the model is over-confident by different amounts on each. A single `T` minimizes
pooled NLL, a population-weighted compromise that can leave a big well-behaved subgroup nicely
calibrated while one minority subgroup stays over-confident and another ends up under-confident, with
the worst-group ECE barely moving. The response, and the reason I put the subgroup-aware method first,
is to fit a separate temperature `T_g` per subgroup, each on its own calibration points by the same NLL
minimization. The richer per-group options — a two-parameter Platt fit per group, per-group binning, a
free per-group monotone map — all multiply the one quantity I am about to show is the killer, so among
per-group families the least-capacity knob, a single temperature per group, is the only one with a
prayer of surviving a shifted tail. Even it needs defending, and that defense is what the variance
calculation below works out.

How well is `T_g` pinned down by `n_g` calibration points? Using `r_g = log T_g` and `q_i(r_g) =
σ(z_i e^{−r_g})`, the per-example NLL curvature near the optimum is the Fisher term
`E[q_i(1−q_i)(z_i e^{−r_g})^2]`, so a group's curvature is `O(n_g)` and `Var(log T_g) ≈ c/n_g`. Make the
constant concrete: a typical over-confident cell has logit magnitude around `|z| ≈ 3`, temperature near
`T ≈ 2` so `e^{−r} = 0.5`, calibrated `q ≈ 0.8` so `q(1−q) ≈ 0.16`, giving per-example Fisher
information `0.16 · (3·0.5)^2 ≈ 0.36`. For a 30-point group total curvature is `≈ 10.8`, so
`Var(log T_g) ≈ 0.093`, a log-temperature standard deviation of `0.30` — the fitted `T` uncertain by a
factor `e^{0.30} ≈ 1.35`, a `±35%` wobble. For a 2000-point group curvature is `720`, `Var ≈ 0.0014`,
standard deviation `0.037`, a `±4%` wobble. The noise falls like `1/√n_g`, and the worst subgroup — the
one I am graded on — is exactly the small cell where it is worst.

And it can be worse than noisy. On a tiny or lopsided sample the NLL can be monotone over the whole
search box, so the fit slams into a boundary. Suppose a group has 15 points, all labeled `1`, all with
scores above `0.5` so every `z > 0`: the NLL `−mean log σ(z/T)` falls monotonically as `T → 0`
(`σ → 1`), there is no interior minimum, and the optimizer walks to `log T = −3`, `T ≈ 0.05`, which
*sharpens* every one of that group's test scores onto the rails — the exact opposite of the softening I
wanted, applied to a shifted tail the sample never represented. So if a group has too few points or only
one class label its local temperature is not merely high-variance but unidentified, and I must refuse to
fit it. Take that noisy or boundary-pinned `T_g` and apply it to the group's *test* points from the
shifted tail: the noise does not average out, it is a per-group systematic distortion on a distribution
the calibration sample never represented, and I would routinely make a small subgroup's calibration
*worse* than if I had used the global `T`. Independent per-group fitting trades a real bias problem for
a worse variance problem, and the worst-group metric reads off exactly the group where that variance is
worst.

This is the statistical shape Stein exposed. I have many related-but-not-identical parameters — one
temperature per group, all "how over-confident is the model here," similar but not equal — and
estimating each in isolation from its own small sample is the coordinatewise MLE, which for
`X ∼ N(θ, σ²I)` in three or more coordinates is inadmissible: another estimator has strictly smaller
total risk for every `θ`, pulling each coordinate toward a common center, harder for the noisier ones
(James & Stein 1961; Efron & Morris 1973 shrank toward the grand mean). The dimension condition holds
here — the subgroups are the cross-product of two protected attributes, four or more cells per dataset,
already past `K ≥ 3`. The common center is sitting right there: the global temperature `T_global` fit on
all the calibration data, the low-variance pooled analogue of the grand mean.

I blend in `θ_g = log T_g`, not `T` directly: `T` is positive and multiplicative, and a convex
combination of log-temperatures exponentiates to a positive temperature automatically, with the two
mirror softenings symmetric about zero. So `log T_g = α_g log T_local,g + (1 − α_g) log T_global`, `α_g`
near 1 for big groups and near 0 for small. The hierarchical model hands me `α_g`: each fit
`m_g = log T_local,g` estimates `θ_g` with sampling variance `σ_w²/n_g`, and the truths scatter about
the center, `θ_g ∼ N(μ = log T_global, σ_b²)`, so the posterior mean is the precision-weighted blend
`θ̂_g = (n_g/(n_g+k)) m_g + (k/(n_g+k)) μ` with `k = σ_w²/σ_b²`. Thus `α_g = n_g/(n_g + k)`, monotone in
group size, with the right limits: `n_g → 0` gives full pooling to global (the local fit told me
nothing), `n_g → ∞` gives no pooling (the local fit is sharp). `k` is the crossover group size and reads
as a prior pseudo-count — the evidence a group must accumulate before it earns its own estimate.

What should `k` be? I could estimate `σ_w²` and `σ_b²` from the data, but with only a handful of groups
the between-group variance is itself a high-variance estimate — the very disease I am treating. Better
to fix `k` conservatively. The subgroup samples run from dozens to a couple thousand points, and given
the shift I want a group to need real evidence before it half-trusts its own temperature, so `k = 200`:
a 30-point group gets `α = 0.13`, a 50-point `0.20`, a 100-point `0.33`, a 200-point the crossover
`0.50`, a 500-point `0.71`, a 2000-point `0.91`. Lined up against the variance computation, the
30-point group whose local temperature was a `±35%` guess keeps only `13%` of its own weight, diluted
roughly seven-to-one by the sharp global estimate, while the 2000-point group accurate to `±4%` keeps
`91%` — the shrinkage pulls the noisier fit home harder, exactly the Stein thing. The degenerate tail
gets the hard guard: fewer than 20 points or all-one-class labels means the local NLL is unidentified,
so I set `T_g = T_global` outright, below where `α` would have given it real weight anyway. Throughout I
clip `p` and `q` into `[ε, 1−ε]` with `ε = 1e-6` and optimize `log T` over `[−3, 3]` (`T ≈ [0.05, 20]`)
so the search stays conditioned and `T` cannot run away on a flat objective. The full module is in the
answer; it degenerates to plain global temperature scaling when no group ids are supplied.

What do I expect from the subgroup-aware method — global temperature, per-group temperatures, and
log-space shrinkage of the small groups toward the global fit? The optimistic read is that on a dataset
with one large, well-determined subgroup
mis-scaled differently from the rest, the per-group temperature (mostly local) drives that group's ECE
down and the worst-group number with it. The pessimistic read, which I half expect, is structural: the
worst subgroup is usually the *small* one, exactly the group `α` pulls toward `T_global`, so on that
group I am effectively doing global temperature scaling with a thin local perturbation — shrinkage
protects me from disaster but denies the small group the individual correction it would need, and the
shift can punish even the global temperature when the calibration-region scale does not transfer to the
tail. So the worst-group ECE may end up barely better than a plain global scalar. There is a clean way
to tell the reads apart: `subgroup_auroc` is a control that must come back at exactly the raw ranking on
each dataset, since every temperature is strictly monotone — if it moves, I have a bug. `brier` is where
the per-group degrees of freedom show up if they help on average. The worst-group ECE is the one
genuinely in doubt, and if `brier` looks healthy while it sits near what a single global scalar would
post, the next move is already written: step *back* to the simplest global scalar, confirm that floor
cleanly, and only then decide whether to climb a different axis — a richer *global* map shape rather
than more per-group parameters.
