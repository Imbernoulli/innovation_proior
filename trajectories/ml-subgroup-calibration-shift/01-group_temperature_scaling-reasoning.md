The research question is about subgroups, so I start by attacking subgroups head-on, and the floor I
build from is the identity fill: clip `p` into `[ε, 1−ε]` and hand it back unchanged. That floor is not
a real method, it is the absence of one — it reports the base classifier's raw probabilities and lets
the metric measure how wrong they are. The reason the task ships a frozen, deliberately miscalibrated
classifier is that the raw `p` is over-confident, so the identity is going to post a high worst-group
ECE on every dataset; the whole point is to design the map that pulls those numbers back onto the
diagonal, and to do it *inside each subgroup*. Since the metric I am graded on is the worst subgroup, my
instinct is that the strongest first move is the one that lets each subgroup get its own correction. Let
me derive that method carefully, because the subgroup machinery is exactly where this can go wrong under
the shift, and I want to see the failure mode coming.

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
divide the logit by a positive number, `q = σ(z/T)`. Stare at what that does. `T = 1` is the identity,
exactly the floor I am replacing. `T > 1` shrinks every logit toward zero, so every `q` moves toward
`1/2` — it *softens*, raises the entropy, which is exactly the cure for over-confidence. `T < 1`
sharpens. As `T → ∞` everything collapses to `1/2`; as `T → 0` everything snaps to a hard 0 or 1. And
here is the property I really need under this benchmark: `z/T` is a monotone increasing function of `z`
for any `T > 0`, so it never changes the order of examples and never moves the `z = 0 ↔ p = 0.5`
decision boundary. The predicted class — and therefore accuracy, and therefore `subgroup_auroc` — is
untouched. One positive scalar, fit to soften the model just enough, with no risk to ranking. This is
temperature scaling, the one-parameter special case of Platt's `q = σ(a·z + b)`: set `a = 1/T` and drop
the intercept `b`. I deliberately drop `b`, because a nonzero intercept makes the boundary `a·z + b = 0`
no longer `z = 0`, which would let the recalibration *change* predictions and disturb the ranking I am
told is good; and because every extra parameter is capacity I will have to pay for in variance, which is
going to matter enormously once I split by subgroup.

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
just `q = σ(z/T)`, `z = logit(p)`. The scalar is the honest minimal fix.

So a global temperature is well-founded, but it is *one* number for *everyone*, and my objective is the
*worst subgroup*, not the average. Different subgroups can genuinely need different amounts of softening
— their score distributions and base rates differ, so the model is over-confident by different amounts
on each. A single `T` minimizes pooled NLL, a population-weighted compromise; it can leave a big
well-behaved subgroup nicely calibrated while one minority subgroup stays over-confident and another
ends up under-confident, and the worst-group ECE barely moves. The obvious response, and the reason I
am putting the subgroup-aware method first, is to stop sharing: fit a separate temperature `T_g` per
subgroup, each on its own calibration points, each by the same NLL minimization. Per-group softening,
each group fixed on its own terms — exactly what a metric over subgroups seems to ask for.

But I can see it does not hold up before I even run it, and naming the failure now is the point of this
rung. How well is `T_g` pinned down by `n_g` calibration points? Use the parameter I actually optimize,
`r_g = log T_g`, and write each fitted probability as `q_i(r_g) = σ(z_i e^{−r_g})`. For one example the
NLL derivative is `(y_i − q_i) z_i e^{−r_g}`; near the optimum the expected curvature is the Fisher term
`E[q_i(1−q_i)(z_i e^{−r_g})^2]`, a per-example constant as long as the group's score distribution is not
degenerate. A group loss is a sum of `n_g` such terms, so its curvature is `O(n_g)`, and the variance
of the fitted `r_g` is the inverse curvature: `Var(log T_g) ≈ c/n_g`. A subgroup with a few thousand
points gets a sharp, trustworthy temperature; a subgroup with thirty points gets a temperature read off
an almost-flat likelihood, and worse, on a small or lopsided sample the NLL can be monotone over the
whole search box, so the "fit" just slams into a boundary — a `T` of 20 or 0.05 that means nothing
except "this group's likelihood did not constrain me." Then I take that noisy, possibly boundary-pinned
`T_g` and apply it to that group's *test* points, drawn from the **shifted** tail. The noise does not
average out; it is a per-group systematic distortion applied to a distribution the calibration sample
never represented. I would routinely make a small subgroup's calibration *worse* than if I had just
used the global `T`. So independent per-group fitting trades a real bias problem (one `T` cannot fit
everyone) for a worse variance problem (each small `T_g` is garbage) — and the worst-group metric reads
off exactly the group where that garbage is worst.

What kind of mistake is "fit each group's parameter from its own sample"? It is the statistical shape
Stein exposed. I have many parameters — one temperature per group — that are *related but not identical*:
they are all "how over-confident is the model here," they should be similar, but not equal. Estimating
each in isolation from its own small noisy sample is the coordinatewise MLE, and for `X ∼ N(θ, σ²I)` in
three or more coordinates that estimator is *inadmissible* — there is another with strictly smaller
total squared-error risk for every `θ`, and it works by pulling every coordinate toward a common center,
harder for the noisier ones (James & Stein 1961; Efron & Morris 1973 made it concrete by shrinking
toward the grand mean). The lesson transfers: do not estimate `K` temperatures independently; pull each
toward a common center, the noisy ones harder. The common center is sitting right there — the global
temperature `T_global` fit on all the calibration data, the low-variance pooled estimate, the analogue
of the grand mean.

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
temperature. `k = 200` does that: a 200-point group is weighted 50/50, a 50-point group gets `α =
50/250 = 0.2` (mostly global), a 2000-point group gets `α ≈ 0.91` (mostly local). And the degenerate
tail needs a hard guard: if a group has fewer than 20 points or all-one-class labels, its local NLL is
not just noisy but unidentified — the minimizer wanders to a box boundary — so I refuse to fit it and
set `T_g = T_global` outright, below the point where `α` would have given it any real weight anyway.
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
exactly the group `α` pulls toward `T_global`, so on that group I am effectively just doing global
temperature scaling — which means the worst-group ECE I post here may end up barely better than (or even
behind) a plain global scalar, because shrinkage protects me from disaster on the small group but also
denies it the individual correction it would need, while the test-time shift can punish even the
*global* temperature when the calibration-region scale does not transfer to the shifted tail. If that is
what the numbers say — if the subgroup-aware method does not clearly beat what a single global scalar
would do, or if its `brier` looks fine while its worst-group ECE does not — then the diagnosis for the
next rung is already written: the per-group degrees of freedom are not buying worst-group calibration
under this shift, so I should step *back* to the simplest global scalar, confirm that floor, and then
climb a different axis — a richer *global* map shape rather than more *per-group* parameters. I will know
which story is true the moment I read the worst-group ECE against the gap and the Brier.
