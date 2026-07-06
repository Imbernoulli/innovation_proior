VICReg landed where I bet it would on two of the three backbones, and the exact shape of the numbers is
what settles the family question. Seed 42 gave 89.85 on ResNet-18, 89.5 on ResNet-34, 91.38 on
ResNet-50 — aggregate (89.85 + 89.5 + 91.38)/3 = 90.24, against barlow's (89.12 + 90.28 + 90.62)/3 =
90.01, a gap of just +0.23. That is a near-tie on the mean, but the per-backbone deltas are much larger
than the aggregate and they do not point the same way: on ResNet-18 VICReg beat barlow by +0.73 (89.85
vs 89.12), on ResNet-50 by +0.76 (91.38 vs 90.62), and on ResNet-34 it *lost* by −0.78 (89.5 vs 90.28).
So a ±0.75-per-backbone swing averages out to a +0.23 tie. Read the ResNet-18 result first, because it
was my explicit bet: the decoupled per-branch geometry in the narrowed 1024-wide projector did transfer
a touch better on the smallest backbone, exactly as I argued the coupling-plus-standardization in barlow
was costing there. Good — that prediction held. But the ResNet-34 number is the one that reorganizes my
thinking. VICReg's own ordering across scale is 91.38 (rn50) > 89.85 (rn18) > 89.5 (rn34): the *middle*
backbone is the odd one out, sitting below even the smallest. Barlow, by contrast, was a clean monotone
with scale (90.62 > 90.28 > 89.12). So the cleaner per-branch method actually *broke* the monotonicity
barlow had, and its spread widened rather than tightened — max−min = 91.38 − 89.5 = 1.88 for VICReg
versus 90.62 − 89.12 = 1.50 for barlow. That is the opposite of what "a cleaner member of the family"
was supposed to buy.

The ResNet-34 dip is precisely the risk I flagged in the VICReg derivation: the heavy `cov_coeff = 100`
in the narrow 1024-dim space over-decorrelating before the representation is fully informative on that
one architecture. I do not get to see per-layer diagnostics, but the signature fits — one backbone
depressed while its neighbours are lifted is what a slightly-too-aggressive decorrelation term looks
like, biting where the interaction with that particular width and depth happens to be worst. So the
deeper lesson from the barlow→VICReg step is not "VICReg beats barlow" — on aggregate they are inside a
quarter-point of each other. It is that two *different* second-moment formulations land within ~0.2 on
the mean while *neither* is stable across all three backbones: barlow is monotone but leaves ResNet-18
lagging by over a point, VICReg lifts ResNet-18 but craters ResNet-34 and goes non-monotone. Both pin
only the embedding's *second moments* — barlow zeros a cross-correlation, VICReg floors per-dimension
variance and zeros per-branch covariance — and a method that controls only the first two moments leaves
the entire rest of the distribution free. That un-pinned tail is exactly where a per-backbone wobble can
live: two architectures can both satisfy "variance floored, dimensions decorrelated" and yet arrive at
embeddings whose third and higher moments, whose actual *shape*, differ enough to transfer differently
under the frozen linear probe. The opening I named at the close of the VICReg derivation is now forced,
and forced by measured non-monotonicity rather than taste: stop pinning second moments and pin the
*whole* embedding distribution to a target.

But "pin the whole distribution" is only a slogan until I say to *what*. Neither barlow nor VICReg ever
derived a target; each picked a statistic — decorrelation, a variance floor — and hoped the resulting
geometry transfers, and the hope held to within a point but wobbled underneath. I would rather derive the
target, because if I know the distribution the embeddings *should* have, I can regularize toward it
directly and stop guessing at coefficients and projector widths per method. So ask the honest question:
at pretraining time I do not know the downstream task — the whole premise is a frozen representation
probed for tasks I have not seen — so I should pick the embedding geometry that is best in the *worst
case* over unknown tasks. That worst-case framing is what turns a matter of taste into a derivation.

Start with the simplest probe, the linear one this harness literally uses, and work out what covariance
shape it prefers. Fit ridge regression to some unknown label direction `y = Zβ + ε` on the frozen
features `Z`. The ridge solution is `β̂ = (ZᵀZ + λI)⁻¹ZᵀY`, so its expectation is
`E[β̂] = (ZᵀZ + λI)⁻¹ZᵀZ β`, and the bias is `E[β̂] − β = (ZᵀZ + λI)⁻¹(ZᵀZ − (ZᵀZ + λI))β =
−λ(ZᵀZ + λI)⁻¹β`. Diagonalize the Gram matrix `ZᵀZ = Σ_j λ_j u_j u_jᵀ`; the bias along eigendirection
`u_j` is `−λ/(λ_j + λ) · β_j`, and its magnitude `λ/(λ_j + λ)` *grows as `λ_j` shrinks*. So the bias is
worst along the *weakest* eigendirection: `λ/(λ_min + λ)`. Compare an anisotropic spectrum against an
isotropic one carrying the same total energy `Σ_j λ_j`; the isotropic case has every `λ_j = λ̄`, so its
worst-direction bias is `λ/(λ̄ + λ)`, and since `λ_min < λ̄` for any spectrum that is not already flat,
the anisotropic representation is strictly worse on *some* downstream direction — the one that happens to
align with its weak eigenvector. Now the variance side, which points the same way: the unregularized
estimator variance is `σ² tr((ZᵀZ)⁻¹) = σ² Σ_j 1/λ_j`, and minimizing `Σ_j 1/λ_j` at fixed total energy
`Σ_j λ_j` is exactly the AM–HM / Jensen problem for the convex function `1/x` — the minimum is at
*equal* eigenvalues. Both the worst-case bias and the total variance are minimized by the same thing: the
embeddings should be **isotropic**. That already explains the wobble I measured — barlow and VICReg push
toward isotropy through their second-moment terms but never *pin* it, so each backbone lands at a
slightly different residual anisotropy, and the probe pays a different bias on each.

Isotropy fixes only the shape of the covariance, though, and a worst-case-over-tasks argument should
care about more than two moments — which is the whole reason second-moment methods can wobble at fixed
covariance. So push the probe from linear to nonlinear — a radius-kNN or kernel estimator, the kind of
head a richer downstream task would use. Its leading integrated squared bias is governed not by the
covariance but by the Fisher-information functional of the embedding density,
`J(p) = ∫ ‖∇ log p‖² p dx`. Which density minimizes `J` at fixed covariance `Σ`? Set up the
Cramér–Rao bound on the location family `p_θ(x) = p(x − θ)`: the identity estimator `T(X) = X` is
unbiased for the location `θ`, so `Cov(T) = Σ ⪰ I(θ)⁻¹`, hence `I(θ) ⪰ Σ⁻¹`, hence
`J(p) = tr I(θ) ≥ tr(Σ⁻¹)`, with equality iff the estimator is efficient, iff the score `∇ log p` is
affine in `x` — iff `p` is **Gaussian**. So among all densities with a given covariance, the Gaussian
uniquely minimizes the functional the nonlinear-probe bias depends on. Now combine the two axioms:
minimizing `tr(Σ⁻¹) = Σ_j 1/λ_j` under a fixed scalar total forces, by the same Jensen argument,
`Σ = sI`. Isotropy and Gaussianity drop out together — the embeddings should be distributed as an
**isotropic Gaussian** `N(0, I)`. That is a target neither barlow nor VICReg has; they enforce
*consequences* of it (decorrelation is "off-diagonals of `Σ` are zero," the variance floor is "diagonal
of `Σ` is bounded below") but never the distribution itself, and crucially they say nothing about the
third and higher moments — the part that is free to differ per backbone. Pinning `N(0, I)` subsumes both
second-moment demands (an isotropic covariance is decorrelated and equal-variance) *and* controls every
higher moment, which is exactly the degrees of freedom the wobble was hiding in.

Now the hard part: how do I push a batch of embeddings toward `N(0, I)` cheaply, and what does the
design space of "match a distribution" actually offer me? The naive instinct is a divergence — KL or
MMD — between the empirical embedding distribution and the target. Walk that a step: a KL to `N(0, I)`
needs a density estimate of the embeddings in D-dimensional space, and D here is in the hundreds to
thousands; density estimation in that dimension is the curse I am precisely trying to avoid, needing
sample counts exponential in D that a batch of 256 comes nowhere near. MMD is kernel-based and dodges
explicit density estimation, but its unbiased estimator is a double sum over all sample pairs, O(B²),
and its power in high dimension degrades unless the kernel bandwidth is tuned to the data — another
sample-hungry, batch-coupled object. So divergences are out for the same reason contrastive learning was
out two rungs ago: they estimate a high-dimensional spread from pairwise sample interactions, and this
batch-256 harness punishes exactly that. The off-the-shelf *multivariate normality tests* —
Baringhaus–Henze, Henze–Zirkler, Mardia — are the same shape, `(1/B²) Σ_i Σ_j k(x_i, x_j)` double sums
that couple every sample to every other at O(B²D) cost. At B = 256 that is 65k pairs of D-vectors per
step, batch-coupled and sample-hungry. I want the opposite: a statistic that is a per-sample *average*,
O(B) not O(B²), so the batch size stops being the bottleneck.

The escape is *slicing*, and it is a theorem, not a heuristic. Cramér–Wold: two random vectors are equal
in distribution iff all of their one-dimensional projections are equal in distribution. So instead of
testing the D-dimensional law directly, I can test the univariate projections `aᵀz` against the
projection of the target — and here the target makes the slicing especially clean. For `z ~ N(0, I)` and
any *unit* vector `a`, the projection `aᵀz` is `N(0, aᵀI a) = N(0, ‖a‖²) = N(0, 1)`: every unit slice of
an isotropic Gaussian is *the same* standard normal, so I never need a per-direction target — each
univariate test is just "are these projected scalars `N(0, 1)`?" A D-dimensional Gaussianity test becomes
a family of identical one-dimensional standard-normality tests, one per random direction, each operating
on a scalar batch at O(B) cost. Sample a finite set of unit directions, run the univariate test on each,
and aggregate. On aggregation I have a real choice: max over directions, or mean. The max routes gradient
only through the single worst-fitting direction — sparse, jumpy, and it moves the target every step as
the worst direction changes. The mean routes gradient through *all* sampled directions at once, a dense
signal, which is what stable optimization wants. So I average. And because any finite set of directions
under-covers the sphere, I resample the directions every step, so the coverage of the sphere *accumulates*
over training rather than being fixed at whatever 256 directions I drew once.

The last choice is which univariate goodness-of-fit statistic to put on each slice, and again the design
space narrows to one member under the constraint that it be differentiable and well-behaved as a loss.
Moment-based tests — skewness, kurtosis — are non-identifiable at any finite order (matching a handful of
moments does not pin the distribution) and their gradients explode at high order, since an `m`-th moment
weights the tails like `x^m`. CDF-based tests — Kolmogorov–Smirnov, Cramér–von Mises, Anderson–Darling —
require *sorting* the samples to form the empirical CDF, and sorting is non-differentiable, so its
gradient is useless for training an encoder. The characteristic-function family threads the needle: the
empirical characteristic function `φ̂(t) = (1/B) Σ_b exp(i t x_b)` is a differentiable *average* over the
batch (O(B), no sort, no pairs), it is *identifiable* because it is the full Fourier content of the
distribution rather than a finite truncation, and — for the Epps–Pulley weighted-L² distance between
`φ̂` and the target CF — it has bounded gradient and curvature regardless of how non-Gaussian the input
currently is. So the per-slice test is Epps–Pulley: project onto a direction, compare the empirical CF of
the projection against the CF of `N(0, 1)` in a Gaussian-weighted L² norm over a frequency grid,
integrate, average over directions, and add it to the invariance term with a single coefficient.

Let me verify the pieces of that statistic actually do what I need before I commit, because this is where
a slick derivation can hide a dead gradient. The CF of `N(0, 1)` is `φ(t) = E[e^{itX}] = exp(−t²/2)`, and
the Epps–Pulley weight is a Gaussian window `exp(−t²/2)` — so, conveniently, the *target CF and the weight
are the same function*, `exp(−t²/2)`, which is exactly what the code carries as `exp_f`. The per-slice
statistic is `∫ exp(−t²/2) · |φ̂(t) − exp(−t²/2)|² dt`. Check the two limits by hand. If the projected
batch is genuinely standard-normal, `φ̂(t) → exp(−t²/2)`, the integrand `→ 0`, and the statistic bottoms
out — the loss floor sits exactly at `N(0, 1)`, as it must. Now the collapse limit, which is the one that
has to bite: if the projection collapses to a near-constant `x_b ≈ c` (zero variance), then
`φ̂(t) = (1/B) Σ_b e^{itx_b} ≈ e^{itc}`, whose modulus is `1` for *every* `t`, while the target
`exp(−t²/2)` decays away from `t = 0`. The squared error `|e^{itc} − exp(−t²/2)|² = 1 − 2cos(tc)exp(−t²/2)
+ exp(−t²)` is zero at `t = 0` but strictly positive elsewhere on the grid, so the Gaussian-weighted
integral over `[−3, 3]` is strictly positive. A collapsed slice is a *high*-loss state. And note what this
buys me that VICReg needed a separate term for: because I test against `N(0, 1)` and not merely "Gaussian
shape," the statistic penalizes a low-variance projection directly — an under-spread slice fails the test
because its CF stays concentrated near 1 while the standard-normal target has already decayed. The single
Gaussianity-vs-`N(0, 1)` test therefore enforces the variance floor, the decorrelation, *and* the higher-
moment shape all at once, per view — the anti-collapse pressure and the distributional target are the
same term, which is exactly the unification the second-moment methods lacked.

Now ground it in *this* harness's edit, because what it runs is the BCS (Batched Characteristic Slicing)
form, a deliberately stripped version of the full machinery, and the simplifications are load-bearing for
both the cost and the numbers. The contract is the same `forward(z1, z2)` → scalar. I hold
`num_slices = 256` random directions, drawn fresh each call from a `torch.Generator` seeded by an
internal `step` counter — so the directions advance over training and the sphere accumulates coverage —
and normalized to unit norm so each projection targets the same `N(0, 1)`. I project each view onto them,
`z @ A`, which is the whole per-step cost besides the CF: `O(B · D · num_slices)`. At B = 256, D = 128,
and 256 slices that is `256 · 128 · 256 ≈ 8.4M` multiply-adds for the projection, and the Epps–Pulley
statistic per slice is a mean over the batch at each of the grid points, `O(B · num_slices · n_points) =
256 · 256 · 10 ≈ 0.66M` — both linear in B, no `B²` pair term anywhere. That is the concrete payoff of
slicing over the multivariate normality tests I rejected: the same batch-256 that starves an O(B²)
double-sum is plenty for a family of O(B) per-slice averages. The Epps–Pulley grid is `t = linspace(−3,
3, 10)`, the complex empirical CF is `ecf = (1j · x_t).exp().mean(0)`, the window `exp(−t²/2)` serves as
both target CF and weight, the weighted squared error is `exp(−t²/2) · |ecf − exp(−t²/2)|²`, and
`torch.trapz` integrates over `t`. The BCS loss is the mean of that over the two views' slices; the
invariance term is `F.mse_loss(z1, z2)`; total is `invariance + lmbd · bcs` with `lmbd = 10.0` — the bcs
statistic is a small weighted-integrated quantity while the MSE invariance is `O(1)`, so the coefficient
of 10 is what lifts the Gaussianity certificate to a magnitude where it actually competes with invariance
rather than being swamped; I cannot pin the exact ratio without seeing the loss traces, but the intent is
that neither term drowns the other.

Three things are *simpler* than the full LeJEPA-style machinery and I should be explicit that the harness
omits them, because they bound how tightly the certificate holds. It integrates the *full* symmetric grid
`[−3, 3]` with `torch.trapz` rather than exploiting evenness to fold it onto `[0, 3]` and halve the knot
budget; it keeps complex arithmetic (`1j`) rather than the algebraically-equal cos/sin-mean real form;
and it seeds directions from a plain per-module `step` counter rather than a cross-device-synchronized
one with an `all_reduce` — single device here, so no synchronization is needed for the slices to be
consistent. It also uses only 10 frequency knots, not the 17 a higher-resolution quadrature would take.
None of these change the algorithm; they trade a little quadrature resolution and DDP-readiness for a
compact single-device implementation. The one configuration choice that *does* change the behaviour is
`CONFIG_OVERRIDES = {"proj_output_dim": 128}`, narrowing the projector output to 128. The reason is a
coverage argument: a Gaussianity-on-random-projections certificate is only as tight as the density of
directions with which I probe the sphere `S^{D−1}`, and covering that sphere to a fixed angular resolution
needs a number of directions that grows *exponentially* in D. A fixed budget of 256 slices per step (plus
the accumulation from resampling) is a far denser cover of `S^{127}` than of `S^{2047}`, so the same
num_slices pins the whole distribution much more tightly at D = 128 than it could at the default 2048.
Fewer output dimensions is also fewer degrees of freedom to certify Gaussian, so the certificate is
sharper for free. This is the reverse of the pressure barlow and VICReg felt: barlow kept the default
2048 and VICReg narrowed to 1024 because their second-moment terms wanted enough width to spread variance
across; SIGReg wants the *narrowest* space, 128, because its anti-collapse term is a distributional
certificate that concentrates as D shrinks. Laid side by side — barlow 2048, VICReg 1024, SIGReg 128 —
the projector width narrows monotonically across the ladder, and it narrows in step with how much of the
distribution each method actually controls.

So the delta from VICReg is precise: where VICReg pinned only the embedding's second moments per branch —
a variance floor plus a covariance decorrelation, leaving every higher moment free, which is where the
per-backbone wobble and the ResNet-34 dip lived — SIGReg pins the *whole* per-view embedding distribution
to the derived `N(0, I)` target by slicing it into 256 one-dimensional Epps–Pulley standard-normality
tests in a deliberately narrow 128-dim space, averaged over directions for dense gradients, added to
plain-MSE invariance with weight 10.

Falsifiable expectations against VICReg's 89.85 / 89.5 / 91.38 (and barlow's 89.12 / 90.28 / 90.62).
Because the target distribution is *derived* from the linear- and nonlinear-probe worst case rather than
guessed, and because pinning the full distribution controls the higher moments the second-moment methods
left free, I expect SIGReg to clear both on aggregate — low-90s on every backbone. But the aggregate is
the weaker prediction; the sharp, more telling one is about *stability across backbones*. The whole
diagnosis of this rung is that the wobble — VICReg's ResNet-34 dip below its own ResNet-18, the broken
monotonicity, the spread widening from 1.50 to 1.88 — is the signature of un-pinned higher moments. If
that diagnosis is right, pinning the full distribution should *tighten* the per-backbone spread and
restore order: I expect all three backbones within roughly half a point of each other, no architecture
dipping below its smaller sibling, and the single largest gain over VICReg landing on ResNet-34 (its
weakest at 89.5), precisely where second-moment pinning was doing worst. That is a prediction that can be
wrong two ways. If SIGReg comes in strong on aggregate but the spread stays wide or ResNet-34 stays
depressed, then the wobble was *not* about higher moments and my whole derivation of the target is
suspect. And if it lands merely *level* with VICReg rather than above, the likely culprit is not the
target but the stripped 10-knot full-grid quadrature in the 128-dim space being too coarse to certify
Gaussianity tightly — a resolution problem, fixable by a finer grid, not a reason to abandon `N(0, I)`.
The `val_acc` table across the three backbones is what tells me which of these it is. (The distilled
module and the literal scaffold edit are in the answer.)
