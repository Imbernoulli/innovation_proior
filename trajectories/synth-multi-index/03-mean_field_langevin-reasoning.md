Two-stage moved the needle but in a way that confirms my worry rather than resolving it. The MSE fell
from the floor's 9–10 to 7.4/7.6/7.2, and the score rose an order of magnitude into the `1e-4` band —
both exactly as predicted, and both attributable to the convex stage-2 ridge solve, which stopped the
readout from inflating variance. But look at `subspace_err`: r2 went 1.95 → 1.81, r3 went 2.37 → 2.30,
r4 went 2.74 → 2.58. That is *barely* below `√(2r)`. On r2 the seed-42 run is still pinned at 1.99 —
identical to the floor — and only seeds 123/456 nudged down (1.56, 1.89). On r3 and r4 the movement is
within a few percent of the uncorrelated value. This is the diagnostic I flagged: if `subspace_err`
stays near `√(2r)` while the MSE moves, stage 1's degree-3 climb found *almost nothing*, and the gain
is essentially the random-feature ridge fit. That is what happened. Freezing the readout and
renormalizing the rows removed the things that were *hurting* joint SGD, but it did not give the first
layer a signal strong enough to actually rotate into `V*`. The moderate-step spherical SGD is still
fighting the same `d²` wall — a decoupled `Σ He₃` with no staircase, each direction needing a degree-3
correlation to rise above the finite-batch sampling floor, and 6400 moderate steps are not enough. The
two-stage win is real but capped exactly where the lower bound says it must be: `E[(g − f̂)²] ≥
E[Var(g | P_U x)]`, and with `P_U` near-random that residual variance is most of `Var[g] = 6`.

So the conclusion is forced: I have to stop asking *gradient descent on the cubic* to find the subspace.
The information-exponent wall is a property of the gradient method, not of the problem — the problem has
an `n ~ d` information floor, while gradient descent pays `d²` because one weight trajectory has to
escape a near-flat equator under an `overlap²` drift. Two ideas, and I want to combine them. First: if a
single descending trajectory is the bottleneck, replace it with a *population* of neurons descending a
*convex* functional, so flatness-at-init no longer traps me — and add diffusion (Langevin noise) so the
population explores the flat region the gradient cannot feel. Second, and sharper for *this* link: the
degree-3 structure that the gradient sees only third-order-weakly can be read off *directly* from a
moment of the data, because Stein's identity converts the Hermite-3 correlation into a low-order tensor
I can estimate in one pass over a large pool. Let me develop both, because this rung uses each for a
different part of the job.

Start with why the population/Langevin lift is the right frame even though I will lean hardest on the
moment estimate. The width-`W` net `f̂(x) = Σ_j a_j σ(⟨w_j, x⟩)` and any norm penalty depend on the
weights only through the empirical measure `μ = (1/W) Σ_j δ_{w_j}` over first-layer neurons. The
prediction is *linear* in `μ` and the squared loss is convex in its argument, so the risk is a *convex*
functional of `μ`: the nonconvexity I fought in weight space was an artifact of pinning the measure to
finitely many atoms. In measure space there are no spurious basins. Each neuron should drift by the
negative gradient of the first variation `J'[μ]`, and adding an entropy regularizer `(1/β) H(μ | τ)`
makes the free energy strictly convex with a unique minimizer — and, crucially, its Wasserstein
gradient flow is *exactly* Langevin dynamics: `dw = −∇J'[μ](w) dt + √(2/β) dB`. The entropy term *is* an
injected Gaussian noise of scale `√(2/β)`. That noise does the saddle-escape the third-order drift
cannot: near the flat equator where the pull is `O(overlap²)` and almost nothing, the diffusion keeps
the population exploring, and the exponent that strangled single-trajectory SGD drops out of the
*statistics* — sample complexity becomes `n ~ d_eff` (the effective dimension, here `~ r` since the
relevant subspace is `r`-dimensional and the input isotropic), independent of the information exponent.
And the abstract objective is just a familiar training rule, because `L2` weight decay *together with*
entropy equals relative entropy to a Gaussian base measure — so the algorithm is noisy weight-decayed
gradient descent, with the Euler–Maruyama noise scale fixed precisely at `√(2·lr/β)`. The harness
exposes exactly this: it stores `noise_std = 1/√β`, and the driver (or my step) multiplies it by
`√(2·lr)` and adds it to the parameters, the discretized Langevin noise. I set `β = 10⁵`, `wd = 10⁻⁴`
on both layers — the KL drift — and keep the rows on the sphere (Riemannian retraction after each step),
because on the positively-curved sphere the log-Sobolev constant is polynomial in `d` rather than
`exp(β)`, the difference between fast and hopeless.

But here is where I must be honest about what *this* harness implements, because it is not the textbook
"run noisy weight-decayed GD for 8000 steps." The budget and the finite dataset make a long Langevin
run on the squared-loss gradient pay the same finite-sample noise that capped two-stage. So I do
something sharper that the Langevin frame *licenses* but that finishes the subspace job in essentially
one shot, then hands the link to a closed-form solve. The key realization is about the drift itself.
For the Hermite-3 link, the population correlation a neuron feels is `E[y · He₃(⟨w, x⟩)]`, and the
gradient of its *square* is the natural feature-learning objective — maximize how much each particle
correlates with the third-order subspace signal. I compute this analytically rather than through autograd
on the squared loss: with `z = x @ ŵᵀ` for unit rows `ŵ`, the correlation is `corr = mean(y · He₃(z))`,
its score-gradient is `3(z² − 1)` (from `He₃'`), and the drift on the rows is `2d · corr · (∇corr)`,
projected onto the sphere's tangent and renormalized — one analytic correlation-Langevin step with the
KL decay and the `√(2·lr)·noise_std` diffusion folded in. This is the single feature-learning move, and
it is the population drift of the *correct* third-order objective, not the third-order-weak gradient of
the squared loss. It nudges the particles, but it is not what carries the subspace recovery — what
carries it is the moment estimate I compute alongside.

That moment estimate is the heart of why this rung should finally crack the subspace. Stein's identity
says `E[x · h(x)] = E[∇h]`, and applied to the cubic link it converts the degree-3 correlation into a
*second-order* object I can estimate directly from the data with no gradient climb at all. Concretely,
form the label-weighted input covariance `M = E[y² · x xᵀ]`. Because `y = g(U*ᵀx)` depends on `x` only
through its `V*`-projection, the directions in which `y²` modulates the input variance are exactly the
teacher directions: `M` is a perturbation of `I_d` whose top eigenspace, after centering, is `V*`. So I
estimate `M` in one chunked pass over a *large* fresh pool — I enlarge `make_dataset` to `n = 64{,}000`
points (still inside the `200k` cap), the fresh-sample regime that makes the empirical `M` track its
population value — symmetrize it, take the top-`r` eigenvectors by `eigh`, and *those are my recovered
subspace directions*, computed without ever paying the `d²` gradient wall. This is the moment method the
information-exponent literature points to as the route to near-optimal `Θ(d)` sample complexity, and the
harness caches these `_CACHED_DIRECTIONS` at dataset-construction time so the readout solve can use them.

With the subspace in hand, the second job — the link — is again a convex closed-form solve, but now on
*good* features instead of near-random ones. I do not trust the noisy first layer to be the feature
map; instead I *install* a feature basis explicitly along the recovered directions. For each of the `r`
estimated directions I place rows at a grid of biases (`linspace(−3, 3)`) and both signs, so the ReLU
features tile each teacher direction with thresholded bumps at a range of offsets — a deliberate
construction of a rich univariate basis along every direction the link reads, which is exactly what is
needed to represent `He₃` on each projection. Then I ridge-fit the readout on `n = 20{,}000` of the
pool: form the post-ReLU design matrix, append a ones column, solve `(ΦᵀΦ + λI)a = Φᵀy` with `λ =
10⁻⁴·n`. This is the same convex move two-stage used, but now the features actually span the cubic on
`V*`, so the residual is no longer floored by `Var(g | P_U x)` with random `U` — it is floored by how
well the bias-grid bumps approximate `He₃`, which with a fine grid is small.

So the rung is, in order: one analytic Hermite-3 correlation-Langevin step on the spherical first-layer
particles (the feature-learning drift of the *correct* third-order objective, with KL decay and the
`√(2·lr/β)` diffusion); a moment estimate `M = E[y²xxᵀ]` whose top-`r` eigenspace recovers `V*`
directly from a large fresh pool, bypassing the gradient wall; an explicit installation of a bias-grid
ReLU feature basis along those recovered directions; and a closed-form ridge solve for the readout. The
Langevin machinery justifies *why* a population method with diffusion escapes the exponent that capped
SGD; the moment estimate is *how* this implementation finishes the subspace in one pass; the ridge solve
fits the link on features that finally span it.

Now the falsifiable expectations against two-stage's numbers, because that is the bar. First and most
important, `subspace_err` should *break away* from the `√(2r)` band that both prior rungs were stuck
near. The moment estimator does not climb a third-order gradient; it reads `V*` off a directly estimated
second-order moment, so on a 64k-point pool it should recover the subspace to a real fraction of its
true directions — I expect `subspace_err` to drop well below two-stage's 1.81/2.30/2.58, into a regime
where the top-`r` eigenspace of `M` is genuinely correlated with `V*` rather than barely perturbed from
random. Second, with the features installed along the *recovered* directions and ridge-fit, `test_mse`
should fall *below* two-stage's 7.2–7.6 — not necessarily to zero, because the moment estimate is
imperfect and the bias-grid basis only approximates `He₃`, and a *missed* direction leaves a full
`6/r`-sized chunk of the variance unexplained. Third, `score = exp(−subspace_err²/r)·exp(−test_mse)`
should jump by *two orders of magnitude* over two-stage's `~1e-4`, into the `1e-2`–`1e-1` band, because
both exponentials improve together once the subspace is genuinely found. I should also expect this to be
*high-variance across seeds*: the moment estimator's eigengap depends on how cleanly `E[y²xxᵀ]`
separates the teacher directions on a finite pool, and an unlucky seed can collapse a direction — so I
would not be surprised to see one seed (say a hard r2 draw) lag while others recover well. If instead
`subspace_err` stays near `√(2r)`, the moment estimate failed and this whole rung reduces to a fancier
random-feature ridge — but the entire point of the `E[y²xxᵀ]` move is that it sees the degree-3
structure the gradient could not, so it should be the first rung to actually find the subspace. The full
module — spherical-particle init, the analytic correlation-Langevin step, the moment-covariance subspace
estimate, the installed bias-grid basis, and the ridge readout — is in the answer.
