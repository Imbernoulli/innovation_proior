Two-stage moved the needle but in a way that confirms my worry rather than resolving it. The MSE fell
from the floor's 9–10 to 7.4/7.6/7.2, and the score rose an order of magnitude into the `1e-4` band —
both exactly as predicted, and both attributable to the convex stage-2 ridge solve, which stopped the
readout from inflating variance. But look at `subspace_err`: r2 went 1.95 → 1.81, r3 went 2.37 → 2.30,
r4 went 2.74 → 2.58. Read those as gaps to `√(2r)` = 2.00/2.449/2.828: the recovery is `9.4%`/`6.1%`/
`8.8%` below the uncorrelated value. That is *barely* below `√(2r)`. And the seed breakdown is the tell:
on r2 the seed-42 run is still pinned at 1.989 — identical to the floor to three figures — and only seeds
123 and 456 nudged down, to 1.556 and 1.888. So it is not that stage 1 reliably found a fraction of the
subspace; it is that on a lucky draw one or two rows crossed the borderline SNR I computed (`≈ 0.83`) and
caught a direction, and on an unlucky draw (seed 42) nothing crossed and the number is the floor
verbatim. The MSE story matches: r2 seed-42, with `subspace_err` pinned at the floor, has `test_mse =
9.11` — barely below its own floor, essentially the de-inflated-random-feature value — while seeds 123
and 456, which moved the subspace, sit at `6.50` and `6.44`, right down against the `Var[g] = 6` line I
drew. This is the diagnostic I flagged fired cleanly: where the subspace moved, the MSE fell to `~6.5`;
where it did not, the MSE stayed near the random-feature floor. The two-stage win is real but capped
exactly where the lower bound said it must be: `E[(g − f̂)²] ≥ E[Var(g | P_U x)]`, and with `P_U`
near-random that residual variance is most of `Var[g] = 6`.

So the conclusion is forced: I have to stop asking *gradient descent on the cubic* to find the subspace.
The information-exponent wall is a property of the gradient *method*, not of the problem — the problem has
an `n ~ d` information floor (a moment estimator can see the degree-3 structure with `O(d)` samples),
while gradient descent pays `d²` because one weight trajectory has to escape a near-flat equator under an
`overlap²` drift. The borderline `0.83` SNR of two-stage is the ceiling of that method: even with the
readout frozen, the rows renormalized, and the pool enlarged, the *gradient* signal is marginal, so I get
marginal, seed-dependent recovery. To break through I need a mechanism that does not read the subspace off
a single descending trajectory. Two ideas, and I want to combine them, because this rung uses each for a
different part of the job.

First, the population/Langevin lift, which is the right *frame* even though I will lean hardest on the
moment estimate for the actual recovery. The width-`W` net `f̂(x) = Σ_j a_j σ(⟨w_j, x⟩)` and any norm
penalty depend on the weights only through the empirical measure `μ = (1/W) Σ_j δ_{w_j}` over first-layer
neurons. The prediction is *linear* in `μ` and the squared loss is convex in its argument, so the risk is
a *convex* functional of `μ`: the nonconvexity I fought in weight space — the spurious basins, the flat
equator — was an artifact of pinning the measure to finitely many atoms. In measure space there are no
spurious minima. Each neuron should drift by the negative gradient of the first variation `J'[μ]`, and
adding an entropy regularizer `(1/β) H(μ | τ)` makes the free energy strictly convex with a unique
minimizer — and, crucially, its Wasserstein gradient flow is *exactly* Langevin dynamics: `dw =
−∇J'[μ](w) dt + √(2/β) dB`. The entropy term *is* an injected Gaussian noise of scale `√(2/β)`. That
noise does the saddle-escape the third-order drift cannot: near the flat equator where the pull is
`O(overlap²)` and almost nothing, the diffusion keeps the population exploring, and the exponent that
strangled single-trajectory SGD drops out of the *statistics* — sample complexity becomes `n ~ d_eff`
(the effective dimension, here `~ r` since the relevant subspace is `r`-dimensional and the input is
isotropic), independent of the information exponent. And the abstract objective is just a familiar
training rule, because `L2` weight decay *together with* entropy equals relative entropy to a Gaussian
base measure — so the algorithm is noisy weight-decayed gradient descent, with the Euler–Maruyama noise
scale fixed precisely at `√(2·lr/β)`. The harness exposes exactly this: it stores `noise_std = 1/√β`, and
the driver (or my step) multiplies it by `√(2·lr)` and adds it to the parameters, the discretized
Langevin noise. I set `β = 10⁵`, so `noise_std = 1/√(10⁵) ≈ 3.16·10⁻³`, and `wd = 10⁻⁴` on both layers —
the KL drift — and keep the rows on the sphere (Riemannian retraction after each step), because on the
positively-curved sphere the log-Sobolev constant is polynomial in `d` rather than `exp(β)`, the
difference between fast and hopeless mixing.

But here is where I must be honest about what *this* harness implements, because it is not the textbook
"run noisy weight-decayed GD for 8000 steps." The budget and the finite dataset make a long Langevin run
on the squared-loss gradient pay the same finite-sample noise that capped two-stage at SNR `0.83`. So I
do something sharper that the Langevin frame *licenses* but that finishes the subspace job in essentially
one shot, then hands the link to a closed-form solve. The single feature-learning move I keep is the
population drift of the *correct* third-order objective, computed analytically rather than through
autograd on the squared loss. For the Hermite-3 link the population correlation a neuron feels is `corr =
E[y · He₃(⟨ŵ, x⟩)]`, and I want to maximize its square — how much each particle correlates with the
third-order subspace signal. With `z = x @ ŵᵀ` for unit rows `ŵ`, `He₃'(z) = 3(z² − 1)` is the
score-gradient, the drift on the rows is `2d · corr · (∇_ŵ corr)` projected onto the sphere's tangent and
renormalized, and the Langevin noise and KL decay fold in on top. This is one analytic
correlation-Langevin step of the *right* objective, not the third-order-weak gradient of the squared
loss. It nudges the particles, but I am clear that it is not what carries the subspace recovery — what
carries it is the moment estimate I compute alongside, and I keep the drift step because it is the honest
Langevin motion the frame prescribes, not because I am counting on one step to solve anything.

That moment estimate is the heart of why this rung should finally crack the subspace, and I should derive
it end to end rather than assert that `E[y²xxᵀ]` "sees the subspace." Stein's identity says `E[x · h(x)] =
E[∇h]`; applied to a degree-3 link it converts the third-order correlation into a *second-order* object I
can estimate directly from data with no gradient climb. Form the label-weighted input covariance `M =
E[y² · x xᵀ]`. Because `y = g(U*ᵀx)` depends on `x` only through `z = U*ᵀx`, write `x = U*z + x_⊥` with
`x_⊥` the orthogonal-complement part, independent of `z` and mean zero. Then `M = E[h(z)(U*z + x_⊥)(U*z +
x_⊥)ᵀ]` with `h(z) = y²`; the cross terms vanish by independence and zero mean, leaving `M = U* A U*ᵀ +
E[h(z)]·E[x_⊥ x_⊥ᵀ]`, where `A = E[h(z) z zᵀ]` is `r×r` and `E[x_⊥x_⊥ᵀ] = I − Π*`. Now compute `A`
concretely for `h = y² = (1/r)(Σ_i He₃(z_i))²`. The diagonal: `A_{kk} = (1/r) E[(Σ_i He₃(z_i)²
+ Σ_{i≠j} He₃(z_i)He₃(z_j)) z_k²]`. The `i = k` self term is `E[He₃(z_k)² z_k²] = E[(z³−3z)² z²] =
E[z⁸ − 6z⁶ + 9z⁴] = 105 − 6·15 + 9·3 = 105 − 90 + 27 = 42`. Each `i ≠ k` term is `E[He₃(z_i)²]·E[z_k²] =
6·1 = 6`, and there are `r − 1` of them. The cross `He₃(z_i)He₃(z_j)` terms are odd in an unpaired
coordinate and vanish. So `A_{kk} = (1/r)(42 + 6(r−1)) = (1/r)(36 + 6r) = 6 + 36/r`. The off-diagonal
`A_{kl}` (`k ≠ l`): the self terms `He₃(z_i)² z_k z_l` are odd in `z_k`, zero; the one surviving cross
term contributes `(1/r)·2·E[He₃(z_k) z_k]·E[He₃(z_l) z_l]`, and `E[He₃(z) z] = E[z⁴ − 3z²] = 3 − 3 = 0`,
so `A_{kl} = 0`. Therefore `A = (6 + 36/r) I_r`, and

`M = 6 I_d + (36/r) Π*`.

That is the whole payoff in one line: `M` is `6·I` on the orthogonal complement and `6 + 36/r` on `V*`,
so its top-`r` eigenspace is *exactly* `span(U*)`, with an eigengap of `36/r` above the bulk. Put the
numbers in: the teacher eigenvalue is `24` (r2), `18` (r3), `15` (r4) sitting above a bulk at `6`, an
eigengap of `18`/`12`/`9`. This is a *huge*, clean spectral signal — nothing like the `0.83`-SNR gradient
whisper. It also predicts the rank ordering: the gap shrinks as `36/r`, so r4 is the hardest for the
estimator, but even its `gap = 9` towers over the finite-sample noise. Let me check that noise: on `n =
64{,}000` fresh points the entrywise sampling error of `M` is `~ std(y² x_k x_l)/√n`; with `E[y²] = 6`
and `y²` fluctuations `O(10)`, that is `O(10)/√64000 ≈ 0.04` per entry, and the top-eigenvalue
perturbation scales like `√(d/n)·‖M‖ ≈ √(128/64000)·24 ≈ 0.045·24 ≈ 1.1` — an order of magnitude below
even the r4 gap of `9`. So on a 64k pool the top-`r` eigenspace of the empirical `M` should track `V*`
tightly, and I recover the subspace *without ever paying the `d²` gradient wall*. This is the moment
method the information-exponent picture points to as the route to near-optimal `Θ(d)` sample complexity.

Before I commit to `M = E[y²xxᵀ]` I should check it against the other moment that encodes `V*`, because
there is a more obvious candidate. The link is odd degree-3, so the *third* moment tensor `T = E[y ·
x⊗x⊗x]` is nonzero — it is proportional to the teacher's Hermite-3 tensor `Σ_i u_i^{⊗3}`, and its
symmetric CP factors are exactly the teacher directions `u_i`. That is an even more direct read of `V*`
than a second moment. But price it out. `T` is a `d×d×d = 128³ ≈ 2.1·10⁶`-entry object; estimating it
needs a pass forming `x⊗x⊗x` per sample (`2·10⁶` multiply-adds each), and recovering the `u_i` from it is
a symmetric tensor decomposition — robust power iteration or ALS, `O(d³ r)` per sweep over many sweeps,
with its own initialization and deflation fragility. Against that, `M = E[y²xxᵀ]` is a `d×d = 16{,}384`-
entry matrix, one chunked pass of rank-1 updates, and a *single* `eigh` at `O(d³) ≈ 2·10⁶` flops — and I
just derived that its eigengap is a clean `36/r` with the teacher subspace sitting as the unambiguous
top-`r` eigenspace, no decomposition heuristics required. The third-moment tensor would recover the same
`V*` but costs two orders more memory and a nonconvex decomposition where the second moment costs one
symmetric eigensolve; with an `O(1)` eigengap already in hand, the extra machinery buys nothing. So I
take `M`, and I take it because the arithmetic — cheaper estimator, closed-form eigensolve, provable
`36/r` gap — dominates the tensor route on every axis that matters here.

Let me verify the `M` formula independently before I build on it, by computing its trace two ways and
demanding they agree. From `M = 6I_d + (36/r)Π*`: `tr(M) = 6d + (36/r)·tr(Π*) = 6d + (36/r)·r = 6d + 36`.
Directly: `tr(M) = tr E[y² xxᵀ] = E[y² ‖x‖²]`. Split `‖x‖² = ‖z‖² + ‖x_⊥‖²` with `z = U*ᵀx` and `x_⊥`
independent; `E[y²‖z‖²] = Σ_k E[h(z) z_k²] = Σ_k A_{kk} = r(6 + 36/r) = 6r + 36`, and `E[y²]E[‖x_⊥‖²] =
6·(d − r)`. Sum: `6r + 36 + 6d − 6r = 6d + 36`. The two computations agree exactly, which pins the `A_{kk}
= 6 + 36/r` diagonal I derived and the `6·(I − Π*)` complement piece — the algebra behind the `36/r`
eigengap is right.

So the implementation: I enlarge `make_dataset` to `n = min(64{,}000, max_train_examples)` fresh Gaussian
points — the fresh-sample regime that makes the empirical `M` track its population value with the noise I
just bounded — estimate `M` in a chunked pass, symmetrize it, take the top-`r` eigenvectors by `eigh`,
and cache those as `_CACHED_DIRECTIONS` at dataset-construction time so the readout solve can use them.
The `64k` cap keeps me well inside the harness's `200k` limit.

With the subspace in hand, the second job — the link — is again a convex closed-form solve, but now on
*good* features instead of near-random ones. I do not trust the noisy first layer to be the feature map;
one analytic Langevin step is not going to tile `V*` with the right thresholds. So instead I *install* a
feature basis explicitly along the recovered directions. For each of the `r` estimated directions I place
rows at a grid of biases (`linspace(−3, 3)`) and both signs, so the ReLU features tile each teacher
direction with thresholded bumps at a range of offsets — a deliberate univariate basis along every
direction the link reads, which is exactly what is needed to represent `He₃` on each projection. The
budget arithmetic: with `W = 256` hidden units and `r` directions I get `256/r` rows per direction — `128`
for r2, `85` for r3, `64` for r4 — split across the two signs, so a bias grid of `~64`/`43`/`32` offsets
per sign spanning `[−3, 3]`, fine enough that the thresholded-ReLU bumps resolve the shape of `He₃(z)` on
`z ∈ [−3, 3]` (which covers `>99.7%` of the standard-Gaussian projection mass). Then I ridge-fit the
readout on `n = 20{,}000` of the pool: form the post-ReLU design matrix, append a ones column, solve
`(ΦᵀΦ + λI)a = Φᵀy` with `λ = 10⁻⁴·n = 2`. This is the same convex move two-stage used, but now the
features actually span the cubic on `V*`, so the residual is no longer floored by `Var(g | P_U x)` with
random `U` — it is floored by how well the bias-grid bumps approximate `He₃`, which with a `~32`–`64`-point
grid is small.

Let me put a number on "small," because "the features span the cubic" is only as good as the grid. A
positive/negative pair of ReLUs at adjacent thresholds `b, b'` is a bump whose superposition builds a
piecewise-linear interpolant of any univariate target along the direction, so the ridge can realize
`He₃(z)` on each recovered direction up to the interpolation error of a piecewise-linear fit on the grid.
That error is `~ (Δb)²·max|He₃''|/8` on each cell, and `He₃''(z) = 6z`, whose magnitude on `z ∈ [−3, 3]`
peaks at `18`. With `~32` offsets per sign on r4 the spacing is `Δb = 6/32 ≈ 0.19`, giving a pointwise
error `~ (0.19)²·18/8 ≈ 0.08`; on r2's finer `~64`-offset grid it is a quarter of that. Square and average
that against `Var[He₃] = 6` and the approximation contributes only a percent or so of residual variance
per direction — negligible next to the `6/r` chunk a *missed* direction would cost. So the bias-grid
basis is not the bottleneck; the moment estimate's fidelity is. The features are good enough that
`test_mse` is set by how many directions `M` recovered, exactly as I want the two jobs cleanly separated.

I should also confirm the analytic drift step does what I claim rather than trust the formula. The drift
is `2d · corr · (∇_ŵ corr)`, and since `∇_ŵ(corr²) = 2·corr·∇_ŵ corr`, the drift is `d · ∇_ŵ(corr²)` —
gradient *ascent* on the squared correlation `corr²`, scaled by the ambient dimension `d`. The code adds
it with a positive `lr` (`weight.add_(drift − wd·w, alpha=lr)`), so the sign is ascent, which is right: I
want each particle to *increase* its degree-3 correlation with the label. The `d` prefactor rescales the
`O(1/d)`-small population drift back to `O(1)` so a single `lr = 5e-2` step actually moves the particle,
and the tangent projection `drift − (drift·ŵ)ŵ` plus renormalization keeps it on the sphere — a genuine
Riemannian correlation-ascent step with the KL decay `−wd·ŵ` and the `√(2·lr)·noise_std ≈ √(0.1)·3.16·
10⁻³ ≈ 10⁻³` Langevin jitter folded in. It is the population feature-learning motion the MFLD frame
prescribes; I lean on the moment estimate for recovery, but the drift is a coherent, sign-correct step,
not a decoration.

So the rung is, in order: one analytic Hermite-3 correlation-Langevin step on the spherical first-layer
particles (the feature-learning drift of the *correct* third-order objective, with KL decay and the
`√(2·lr)·noise_std` diffusion); a moment estimate `M = E[y²xxᵀ] = 6I + (36/r)Π*` whose top-`r` eigenspace
recovers `V*` directly from a large fresh pool, bypassing the gradient wall; an explicit installation of a
bias-grid ReLU feature basis along those recovered directions; and a closed-form ridge solve for the
readout. The Langevin machinery justifies *why* a population method with diffusion escapes the exponent
that capped SGD; the moment estimate is *how* this implementation finishes the subspace in one pass; the
ridge solve fits the link on features that finally span it.

Now the falsifiable expectations against two-stage's numbers, because that is the bar. First and most
important, `subspace_err` should *break away* from the `√(2r)` band that both prior rungs were stuck near.
The moment estimator does not climb a third-order gradient; it reads `V*` off a directly estimated
second-order moment whose eigengap `36/r` is `18`/`12`/`9` against a finite-sample perturbation `~1`, so
on a 64k pool it should recover the subspace to a real fraction of its true directions. I expect
`subspace_err` to drop well below two-stage's 1.81/2.30/2.58, into a regime where the top-`r` eigenspace
of `M` is genuinely correlated with `V*` rather than barely perturbed from random. Second, with the
features installed along the *recovered* directions and ridge-fit, `test_mse` should fall *below*
two-stage's 7.2–7.6 — not necessarily to zero, because the moment estimate is imperfect and the bias-grid
basis only approximates `He₃`, and a *missed* direction leaves a full `6/r`-sized chunk (`3`/`2`/`1.5` of
variance for r2/r3/r4) unexplained. Third, `score = exp(−subspace_err²/r)·exp(−test_mse)` should jump by
*two orders of magnitude* over two-stage's `~10⁻⁴`, into the `10⁻²`–`10⁻¹` band, because both exponentials
improve together once the subspace is genuinely found — `exp(−subspace_err²/r)` climbing off `exp(−2)`
toward `1` as the projector distance collapses, and `exp(−test_mse)` climbing off `exp(−7)` toward
`exp(−3)` or better. I should also expect this to be *high-variance across seeds*: the moment estimator's
eigengap `36/r` is population-clean, but on a finite pool an unlucky draw can shrink the *empirical* gap
for one near-degenerate teacher direction and collapse it into the bulk — so I would not be surprised to
see one seed lag while the others recover well, the finite-sample analogue of the seed-42 pinning I saw
in two-stage but now the exception rather than the rule. If instead `subspace_err` stays near `√(2r)`,
the moment estimate failed and this whole rung reduces to a fancier random-feature ridge — but the entire
point of the `E[y²xxᵀ]` move is that its eigengap is `36/r`, an `O(1)` spectral signal rather than an
`O(1/√d)` gradient one, so it should be the first rung to actually find the subspace. The full module —
spherical-particle init, the analytic correlation-Langevin step, the moment-covariance subspace estimate,
the installed bias-grid basis, and the ridge readout — is in the answer.
