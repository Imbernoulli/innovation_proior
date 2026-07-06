The research question is about subgroups, so I start by attacking subgroups head-on, and the floor I
build from is the identity fill: clip `p` into `[ε, 1−ε]` and hand it back unchanged. That floor is not
a real method, it is the absence of one — it reports the base classifier's raw probabilities and lets
the metric measure how wrong they are. The reason the task ships a frozen, deliberately miscalibrated
classifier is that the raw `p` is over-confident, so the identity is going to post a high worst-group
ECE on every dataset. I can already say roughly why the number will be high: the ECE bins confidence
into 15 equal-width buckets and reads the gap between mean confidence and empirical accuracy in each.
An over-confident model dumps most of its mass into the top buckets — the `0.9`–`1.0` bin — where the
reported confidence is near `0.95` but the true positive rate is well below that, so those buckets
carry a large accuracy-minus-confidence gap, and the worst subgroup carries the largest of all. The
whole point is to design the map that pulls those numbers back onto the diagonal, and to do it *inside
each subgroup*. Since the metric I am graded on is the worst subgroup, my instinct is that the strongest
first move is the one that lets each subgroup get its own correction. Let me derive that method
carefully, because the subgroup machinery is exactly where this can go wrong under the shift, and I want
to see the failure mode coming before I run anything.

First, why is `p` wrong, and which way? It is the standard story for a model trained to minimize
log-loss. Once the classifier is getting almost everything right, the loss can still go down by shoving
the probabilities harder toward 0 and 1 — it keeps sharpening past the point where the numbers reflect
real frequencies, overfitting the log-loss long after the 0/1 error has flattened, and all that excess
goes into confidence. The useful part of that observation is *which* way it is wrong: the cheap,
dominant failure is that the overall *scale* of the score is too big — the model is uniformly too sure —
not that the ranking is scrambled. If the ordering were broken I would need something that can reorder
examples; but it is not, the base pipeline is a logistic regression that ranks fine (the diagnostic
`subgroup_auroc` confirms the ranking is intact and a monotone calibrator cannot touch it). So I should
look for the smallest knob that fixes a scale.

Where does "scale" live? Not in `p` directly — `p` is squashed into `[0, 1]` and a "scale" there is not
even well defined. It lives in the logit. For a binary model `p = σ(z)` with `z = logit(p) =
log(p/(1−p))`, and `σ` and `logit` are the canonical monotone bijection between `[0, 1]` and the whole
real line. So any monotone reshaping of `p` is a reshaping of `z`, and the cleanest scale knob is to
divide the logit by a positive number, `q = σ(z/T)`. Let me not take on faith what that does — let me
push three representative over-confident scores through it. Take `p ∈ {0.5, 0.9, 0.99}`, whose logits
are `z = 0, 2.197, 4.595`, and set `T = 2`. Then `z/T = 0, 1.099, 2.298`, and `q = σ(z/T) = 0.500,
0.750, 0.909`. Read that off: the `0.5` example does not move at all, because its logit is zero and
zero divided by anything is zero; the `0.9` is softened to `0.75`; the `0.99` is softened to `0.909`.
The order is preserved (`0.5 < 0.75 < 0.909`), the decision boundary at `p = 0.5` is nailed in place,
and the confidences are pulled toward the middle by an amount that grows with how extreme they were.
`T = 1` is the identity, exactly the floor I am replacing. `T > 1` shrinks every logit toward zero, so
every `q` moves toward `1/2` — it *softens*, raises the entropy, which is exactly the cure for
over-confidence. `T < 1` sharpens. As `T → ∞` everything collapses to `1/2`; as `T → 0` everything snaps
to a hard 0 or 1. And here is the property I really need under this benchmark: `z/T` is a monotone
increasing function of `z` for any `T > 0`, so it never changes the order of examples and never moves
the `z = 0 ↔ p = 0.5` decision boundary. The predicted class — and therefore accuracy, and therefore
`subgroup_auroc` — is untouched. One positive scalar, fit to soften the model just enough, with no risk
to ranking. This is temperature scaling, the one-parameter special case of Platt's `q = σ(a·z + b)`: set
`a = 1/T` and drop the intercept `b`. I deliberately drop `b`, because a nonzero intercept makes the
boundary `a·z + b = 0` no longer `z = 0`, which would let the recalibration *change* predictions and
disturb the ranking I am told is good; and because every extra parameter is capacity I will have to pay
for in variance, which is going to matter enormously once I split by subgroup.

I want to be sure dividing the logit by a scalar is the *right* correction and not just a convenient
one, so let me pin down what it optimizes. I fit `T` by minimizing the calibration-split NLL — binary
cross-entropy `−mean(y log q + (1−y) log(1−q))` — because NLL is a proper scoring rule: in expectation
it is minimized exactly when the reported probability is the true conditional probability, so descending
it literally pushes `q` toward calibration. The ECE I am graded on is the binned accuracy-minus-
confidence gap; it bins and is not differentiable, so it cannot be the thing I fit — I fit NLL and
*measure* ECE. Now, what family of recalibrations does "scale the logits" correspond to? Ask the dual
question: among all per-example distributions `q_i` that are valid probabilities and that match one
moment of the data — the average true-class logit equals the average expected logit under `q` — which
has maximum entropy? Set up the Lagrangian, `L = −Σ_i Σ_k q_i^(k) log q_i^(k) + λ Σ_i [Σ_k z_i^(k)
q_i^(k) − z_i^(y_i)] + Σ_i β_i (Σ_k q_i^(k) − 1)`. Differentiate in `q_i^(k)`: `−log q_i^(k) − 1 + λ
z_i^(k) + β_i = 0`, so `q_i^(k) = exp(λ z_i^(k) + β_i − 1)`, and normalizing over `k` kills `β_i` and
leaves `q_i^(k) = softmax(λ z_i)^(k)`. Write `λ = 1/T` and that is exactly logits-over-`T`. So softening
by a single temperature is not an arbitrary squash — it is the maximum-entropy distribution that
corrects nothing but the average logit scale, precisely the failure I diagnosed. In binary form it is
just `q = σ(z/T)`, `z = logit(p)`. The scalar is the honest minimal fix, and it carries essentially no
capacity to overfit the calibration split, which is the property I will most want to keep as I make the
method richer.

So a global temperature is well-founded, but it is *one* number for *everyone*, and my objective is the
*worst subgroup*, not the average. Different subgroups can genuinely need different amounts of softening
— their score distributions and base rates differ, so the model is over-confident by different amounts
on each. A single `T` minimizes pooled NLL, a population-weighted compromise; it can leave a big
well-behaved subgroup nicely calibrated while one minority subgroup stays over-confident and another
ends up under-confident, and the worst-group ECE barely moves. The obvious response, and the reason I
am putting the subgroup-aware method first, is to stop sharing: fit a separate temperature `T_g` per
subgroup, each on its own calibration points, each by the same NLL minimization. Per-group softening,
each group fixed on its own terms — exactly what a metric over subgroups seems to ask for.

Before I commit to per-group *temperature*, though, let me be honest that temperature is only the
cheapest of several ways to give each group its own map, and weigh the richer ones on their arithmetic
rather than dismissing them by reflex. I could give each group its own two-parameter Platt fit; I could
bin each group separately and report per-bin empirical rates; I could fit a free monotone map inside
each group. Count what each costs on the small cells I am going to face. Platt per group is `2K`
parameters instead of `K`, and the intercept reintroduces exactly the boundary move I just refused, so
it doubles the per-group variance I am already worried about and puts `subgroup_auroc` at risk.
Per-group equal-width binning needs enough points per bin to estimate a rate: 15 bins over a 30-point
group leaves two points per bin on average and most bins empty, so the "calibration" it reports is pure
sampling noise. A free per-group monotone fit has as many effective degrees of freedom as the group has
points, which on a 30-point cell is a memorized interpolation of 30 coin flips. Every richer per-group
family multiplies the one quantity I am about to show is the killer. So among per-group options, the
least-capacity knob — a single temperature per group — is the only one with any prayer of surviving a
shifted tail, and even it needs defending. That defense is the whole content of this rung.

Here is the trouble, and naming it now is the point. How well is `T_g` pinned down by `n_g` calibration
points? Use the parameter I actually optimize, `r_g = log T_g`, and write each fitted probability as
`q_i(r_g) = σ(z_i e^{−r_g})`. For one example the NLL derivative is `(y_i − q_i) z_i e^{−r_g}`; near the
optimum the expected curvature is the Fisher term `E[q_i(1−q_i)(z_i e^{−r_g})^2]`, a per-example
constant as long as the group's score distribution is not degenerate. A group loss is a sum of `n_g`
such terms, so its curvature is `O(n_g)`, and the variance of the fitted `r_g` is the inverse curvature:
`Var(log T_g) ≈ c/n_g`. Let me make that constant concrete rather than leaving it as a symbol, because
the whole argument turns on how fast the variance falls with `n_g`. Take a typical over-confident cell
with logit magnitude around `|z| ≈ 3`, a temperature near `T ≈ 2` so `e^{−r} = 0.5`, and calibrated
probabilities around `q ≈ 0.8` so `q(1−q) ≈ 0.16`. The per-example Fisher information is then
`0.16 · (3·0.5)^2 = 0.16 · 2.25 ≈ 0.36`. For a 30-point group the total curvature is `30 · 0.36 ≈
10.8`, so `Var(log T_g) ≈ 1/10.8 ≈ 0.093`, a standard deviation of `0.30` in log-temperature — meaning
the fitted `T` is uncertain by a multiplicative factor of `e^{0.30} ≈ 1.35`, a `±35%` wobble. For a
2000-point group the curvature is `720`, `Var ≈ 0.0014`, standard deviation `0.037`, a factor of
`e^{0.037} ≈ 1.04`, a `±4%` wobble. So the small cell's temperature is a `±35%` guess and the large
cell's is nailed to `±4%`; the noise falls like `1/√n_g`, and the worst subgroup — the one I am graded
on — is exactly the small cell where it is worst.

And it can be worse than noisy. On a tiny or lopsided sample the NLL can be monotone over the whole
search box, so the "fit" just slams into a boundary — a `T` of 20 or 0.05 that means nothing except
"this group's likelihood did not constrain me." Trace the pathological case explicitly so I know to
guard it. Suppose a group has 15 points, all labeled `1`, all with scores above `0.5` so every `z > 0`.
The group NLL is `−mean log σ(z/T)`, and as `T → 0` the argument `z/T → +∞`, `σ → 1`, and the loss
falls monotonically toward zero. There is no interior minimum: the optimizer walks straight to the lower
edge of the box, `log T = −3`, `T ≈ 0.05`, which *sharpens* every one of that group's test scores onto
the rails — the exact opposite of the softening I wanted, applied to a shifted tail the sample never
represented. That is why a hard guard is not optional: if a group has too few points or only one class
label, its local temperature is not merely high-variance, it is unidentified, and I must refuse to fit
it. Now I take the noisy (or boundary-pinned) `T_g` and apply it to that group's *test* points, drawn
from the **shifted** tail. The noise does not average out; it is a per-group systematic distortion
applied to a distribution the calibration sample never represented. I would routinely make a small
subgroup's calibration *worse* than if I had just used the global `T`. So independent per-group fitting
trades a real bias problem (one `T` cannot fit everyone) for a worse variance problem (each small `T_g`
is garbage) — and the worst-group metric reads off exactly the group where that garbage is worst.

What kind of mistake is "fit each group's parameter from its own sample"? It is the statistical shape
Stein exposed. I have many parameters — one temperature per group — that are *related but not
identical*: they are all "how over-confident is the model here," they should be similar, but not equal.
Estimating each in isolation from its own small noisy sample is the coordinatewise MLE, and for `X ∼
N(θ, σ²I)` in three or more coordinates that estimator is *inadmissible* — there is another with
strictly smaller total squared-error risk for every `θ`, and it works by pulling every coordinate toward
a common center, harder for the noisier ones (James & Stein 1961; Efron & Morris 1973 made it concrete
by shrinking toward the grand mean). The dimension condition matters and I should check I am actually in
it: the subgroups here are the cross-product of two protected attributes — Adult's sex×race, COMPAS's
race×sex, Law School's race×gender — which is a handful of cells, four or more per dataset. Four is
already past the `K ≥ 3` threshold where the coordinatewise MLE becomes inadmissible, so Stein shrinkage
is genuinely available to me and not a large-`K` fantasy. The lesson transfers: do not estimate `K`
temperatures independently; pull each toward a common center, the noisy ones harder. The common center
is sitting right there — the global temperature `T_global` fit on all the calibration data, the
low-variance pooled estimate, the analogue of the grand mean.

In what space do I blend? Not on `T` directly: `T` is positive and multiplicative, and a linear average
weights `T = 4` vs `1` asymmetrically against `1` vs `1/4`, mirror-image softenings, and I would have to
keep checking positivity. The natural coordinate is the one I optimize in, `θ_g = log T_g`: a convex
combination of log-temperatures exponentiates to a positive temperature automatically, and the two
mirror softenings sit symmetrically about zero. So `log T_g = α_g log T_local,g + (1 − α_g) log
T_global`, with `α_g` near 1 for big groups and near 0 for small ones. I do not want to guess `α_g`; the
hierarchical model hands it to me. Each group's fit `m_g = log T_local,g` estimates the truth `θ_g` with
sampling variance `σ_w²/n_g` (the `1/n_g` from the curvature, `σ_w²` the per-point Fisher constant), and
the truths scatter about the center, `θ_g ∼ N(μ = log T_global, σ_b²)`. The posterior mean is the
precision-weighted blend: local precision `n_g/σ_w²`, prior precision `1/σ_b²`, giving `θ̂_g = [(n_g/σ_w²)
m_g + (1/σ_b²) μ] / [n_g/σ_w² + 1/σ_b²]`. Multiply through by `σ_w²` and call `k = σ_w²/σ_b²`: `θ̂_g =
(n_g/(n_g+k)) m_g + (k/(n_g+k)) μ`, so `α_g = n_g/(n_g + k)`. It is monotone in group size, lives in
`(0,1)`, and the limits are exactly right: `n_g → 0` gives `α_g → 0` (full pooling to global, correct
because the local fit told me nothing) and `n_g → ∞` gives `α_g → 1` (no pooling, correct because the
local fit is now sharp). `k` is the crossover group size — at `n_g = k` the group is half local, half
global — and it has a clean beta-binomial reading as a prior pseudo-count: the amount of evidence a
group must accumulate before it earns its own estimate.

What should `k` be? In principle I could estimate `σ_w²` and `σ_b²` and let the data set the shrinkage,
but the number of groups here is tiny — the cross-product of two protected attributes, a handful of
groups — so estimating the *between*-group variance from those few groups is itself a high-variance
estimate, the very disease I am treating. Better to fix `k` conservatively to the regime. The subgroup
calibration samples run from dozens to maybe a couple thousand points, and given the shift I want a
group to need real evidence — on the order of a couple hundred points — before it half-trusts its own
temperature. `k = 200` does that, and it is worth reading the whole shrinkage schedule off it so I know
what the small cells actually get: a 30-point group gets `α = 30/230 = 0.13`, a 50-point group `α =
50/250 = 0.20`, a 100-point group `α = 0.33`, a 200-point group the crossover `0.50`, a 500-point group
`0.71`, and a 2000-point group `α ≈ 0.91`. Now line that up against the variance computation: the
30-point group whose local temperature was a `±35%` guess is granted only `13%` of its own weight, so
its garbage is diluted roughly seven-to-one by the sharp global estimate, while the 2000-point group
whose local fit was accurate to `±4%` keeps `91%` of it. The shrinkage is doing precisely the Stein
thing — the noisier the local fit, the harder `α` pulls it home. And the degenerate tail needs the hard
guard I traced above: if a group has fewer than 20 points or all-one-class labels, its local NLL is not
just noisy but unidentified — the minimizer wanders to a box boundary — so I refuse to fit it and set
`T_g = T_global` outright, below the point where `α` would have given it any real weight anyway.
Throughout I clip `p` and `q` into `[ε, 1−ε]` with `ε = 1e-6` so the logit and the log stay finite, and
I optimize `log T` over `[−3, 3]` (`T ≈ [0.05, 20]`) so the 1-D search stays well conditioned and `T`
cannot run away on a flat objective. The full scaffold module is in the answer; it degenerates to plain
global temperature scaling when no group ids are supplied.

So this rung is the subgroup-aware method built as carefully as I can build it: a global temperature as
the maximum-entropy scale correction, per-group temperatures to chase the worst subgroup, and
empirical-Bayes shrinkage in log-space to keep the small groups from overfitting under the shift. Now
let me reason about what I expect, because the whole reason I am running this first is to learn whether
the subgroup machinery pays off at all on a shifted benchmark. The optimistic read is that on a dataset
with one large, well-determined subgroup that is mis-scaled differently from the rest, the per-group
temperature (mostly local, `α` near 1) drives that group's ECE down and the worst-group number with it.
The pessimistic read, which I half expect, is structural: the worst subgroup is usually the *small* one,
exactly the group `α` pulls toward `T_global` — and I just quantified that a 30-to-50-point cell keeps
only `13`–`20%` of its own temperature, so on that group I am effectively doing global temperature
scaling with a thin local perturbation. Which means the worst-group ECE I post here may end up barely
better than (or even behind) a plain global scalar, because shrinkage protects me from disaster on the
small group but also denies it the individual correction it would need, while the test-time shift can
punish even the *global* temperature when the calibration-region scale does not transfer to the shifted
tail. There is a clean way to tell the two reads apart in the numbers I will get back. The `subgroup_
auroc` column is a control: every temperature is strictly monotone, so AUROC must come back exactly at
whatever the raw ranking gives on each dataset, and if it moves at all I have a bug. The `brier` column
is where the per-group degrees of freedom should show up if they help *on average* — Brier is a proper
score and rewards the population-weighted improvement the local fits on the big cells buy. The worst-
group ECE is the one genuinely in doubt. If `brier` looks healthy while the worst-group ECE sits near
what a single global scalar would plausibly post — up near `0.45`–`0.50` on the harder datasets — then
the diagnosis for the next rung is already written: the per-group degrees of freedom are buying Brier
but not worst-group calibration under this shift, so I should step *back* to the simplest global scalar,
confirm that floor cleanly, and only then decide whether to climb a different axis — a richer *global*
map shape rather than more *per-group* parameters. I will know which story is true the moment I read the
worst-group ECE against the gap and the Brier.
