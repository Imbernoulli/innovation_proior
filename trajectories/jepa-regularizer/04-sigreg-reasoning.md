VICReg landed where I bet it would on two of the three backbones, and the exact shape of the numbers
settles the family question. Seed 42: 89.85 on ResNet-18, 89.5 on ResNet-34, 91.38 on ResNet-50,
aggregate (89.85 + 89.5 + 91.38)/3 = 90.24 against barlow's 90.01 — a +0.23 near-tie on the mean. But the
per-backbone deltas are much larger than the aggregate and do not point the same way: on ResNet-18
VICReg beat barlow by +0.73 (89.85 vs 89.12), on ResNet-50 by +0.76 (91.38 vs 90.62), and on ResNet-34
it *lost* by −0.78 (89.5 vs 90.28). A ±0.75-per-backbone swing averaging to +0.23. The ResNet-18 result
was my explicit bet — the decoupled per-branch geometry in the narrowed 1024-wide projector did transfer
a touch better on the smallest backbone, exactly as I argued the coupling-plus-standardization in barlow
was costing there. But the ResNet-34 number reorganizes my thinking: VICReg's own ordering across scale
is 91.38 (rn50) > 89.85 (rn18) > 89.5 (rn34) — the *middle* backbone below even the smallest. Barlow was
a clean monotone (90.62 > 90.28 > 89.12). So the cleaner per-branch method actually *broke* the
monotonicity barlow had, and its spread widened rather than tightened: max−min = 91.38 − 89.5 = 1.88 for
VICReg versus 90.62 − 89.12 = 1.50 for barlow. The opposite of what "a cleaner member of the family" was
supposed to buy.

The ResNet-34 dip is precisely the risk I flagged in the VICReg derivation: the heavy `cov_coeff = 100`
in the narrow 1024-dim space over-decorrelating before the representation is fully informative on that
one architecture. One backbone depressed while its neighbours are lifted is what a slightly-too-aggressive
decorrelation term looks like, biting where the interaction with that particular width and depth is
worst. So the deeper lesson from barlow→VICReg is not "VICReg beats barlow" — on aggregate they are
within a quarter-point. It is that two *different* second-moment formulations land within ~0.2 while
*neither* is stable across all three backbones: barlow monotone but leaving ResNet-18 lagging by over a
point, VICReg lifting ResNet-18 but cratering ResNet-34 and going non-monotone. Both pin only the
embedding's *second moments* — barlow zeros a cross-correlation, VICReg floors per-dimension variance and
zeros per-branch covariance — and a method controlling only the first two moments leaves the entire rest
of the distribution free. That un-pinned tail is where a per-backbone wobble can live: two architectures
can both satisfy "variance floored, dimensions decorrelated" and yet arrive at embeddings whose third and
higher moments differ enough to transfer differently under the frozen linear probe. So the opening is
forced, by measured non-monotonicity rather than taste: stop pinning second moments and pin the *whole*
embedding distribution to a target.

"Pin the whole distribution" is a slogan until I say to *what*. Neither barlow nor VICReg derived a
target; each picked a statistic and hoped the resulting geometry transfers, and the hope held to within a
point but wobbled underneath. I would rather derive the target: if I know the distribution the embeddings
*should* have, I can regularize toward it directly and stop guessing coefficients and projector widths per
method. At pretraining time I do not know the downstream task — the whole premise is a frozen
representation probed for tasks I have not seen — so I should pick the embedding geometry best in the
*worst case* over unknown tasks. That framing turns a matter of taste into a derivation.

Start with the linear probe this harness literally uses. Fit ridge to some unknown label direction
`y = Zβ + ε`; the solution `β̂ = (ZᵀZ + λI)⁻¹ZᵀY` has bias `E[β̂] − β = −λ(ZᵀZ + λI)⁻¹β`, and diagonalizing
`ZᵀZ = Σ_j λ_j u_j u_jᵀ` the bias along `u_j` is `−λ/(λ_j + λ)·β_j`, whose magnitude `λ/(λ_j + λ)` *grows
as `λ_j` shrinks* — worst along the weakest eigendirection. Against an isotropic spectrum of the same
total energy (`λ_j = λ̄`), the worst-direction bias is `λ/(λ̄ + λ)`, and since `λ_min < λ̄` for any
non-flat spectrum, the anisotropic representation is strictly worse on *some* downstream direction. The
variance side points the same way: the estimator variance `σ² Σ_j 1/λ_j` at fixed total energy
`Σ_j λ_j` is minimized (AM–HM / Jensen for the convex `1/x`) at equal eigenvalues. Both worst-case bias
and total variance are minimized by the same thing: the embeddings should be **isotropic**. That already
explains the wobble — barlow and VICReg push toward isotropy through their second-moment terms but never
*pin* it, so each backbone lands at a slightly different residual anisotropy and the probe pays a
different bias on each.

Isotropy fixes only the shape of the covariance, and a worst-case argument should care about more than
two moments — which is the whole reason second-moment methods can wobble at fixed covariance. So push the
probe from linear to nonlinear — a kernel or radius-kNN head, the kind a richer task would use. Its
leading integrated squared bias is governed by the Fisher-information functional
`J(p) = ∫ ‖∇ log p‖² p dx`. Which density minimizes `J` at fixed covariance `Σ`? The Cramér–Rao bound on
the location family `p_θ(x) = p(x − θ)`: the estimator `T(X) = X` is unbiased for `θ`, so `Σ ⪰ I(θ)⁻¹`,
hence `J(p) = tr I(θ) ≥ tr(Σ⁻¹)`, with equality iff the score `∇ log p` is affine in `x` — iff `p` is
**Gaussian**. Now combine the two: minimizing `tr(Σ⁻¹) = Σ_j 1/λ_j` under a fixed scalar total forces, by
the same Jensen argument, `Σ = sI`. Isotropy and Gaussianity drop out together — the embeddings should be
an **isotropic Gaussian** `N(0, I)`. That is a target neither barlow nor VICReg has; they enforce
*consequences* of it (decorrelation is "off-diagonals of `Σ` are zero," the variance floor is "diagonal
of `Σ` is bounded below") but never the distribution itself, and say nothing about the third and higher
moments — exactly the degrees of freedom the wobble hides in. Pinning `N(0, I)` subsumes both
second-moment demands *and* controls every higher moment.

Now how to push a batch of embeddings toward `N(0, I)` cheaply. The naive instinct is a divergence — KL
needs a density estimate in D-dimensional space (the curse I am trying to avoid, sample counts
exponential in D that a batch of 256 comes nowhere near), and MMD dodges explicit density but is an O(B²)
double sum with a bandwidth to tune. Divergences are out for the same reason contrastive learning was out
two steps ago: they estimate high-dimensional spread from pairwise sample interactions, and batch-256
punishes exactly that. The off-the-shelf multivariate normality tests — Baringhaus–Henze, Henze–Zirkler,
Mardia — are the same O(B²D) double-sum shape. I want the opposite: a per-sample *average*, O(B) not
O(B²). The escape is *slicing*, and it is a theorem. Cramér–Wold: two random vectors are equal in
distribution iff all their one-dimensional projections are. So test the univariate projections `aᵀz`
against the target's — and here the target makes it clean: for `z ~ N(0, I)` and any *unit* vector `a`,
`aᵀz ~ N(0, ‖a‖²) = N(0, 1)`, so every unit slice of an isotropic Gaussian is the *same* standard normal
and I never need a per-direction target. A D-dimensional Gaussianity test becomes a family of identical
one-dimensional standard-normality tests, one per random direction, each on a scalar batch at O(B).
Sample unit directions, run the univariate test on each, aggregate. On aggregation: the max routes
gradient only through the single worst slice — sparse, jumpy, and it moves the target every step as the
worst direction changes — while the mean routes gradient through all sampled directions at once, a dense
signal stable optimization wants, so I average. And because any finite set under-covers the sphere, I
resample directions every step so coverage *accumulates* over training.

Which univariate goodness-of-fit statistic per slice? Moment-based tests — skewness, kurtosis — are
non-identifiable at finite order and their gradients explode, an `m`-th moment weighting the tails like
`x^m`. CDF-based tests — Kolmogorov–Smirnov, Cramér–von Mises, Anderson–Darling — require *sorting* to
form the empirical CDF, and sorting is non-differentiable, so its gradient is useless for training an
encoder. The characteristic-function family threads the needle: the empirical CF
`φ̂(t) = (1/B) Σ_b exp(i t x_b)` is a differentiable *average* over the batch (O(B), no sort, no pairs),
it is *identifiable* — the full Fourier content of the distribution, not a truncation — and for the
Epps–Pulley weighted-L² distance between `φ̂` and the target CF it has bounded gradient and curvature
regardless of how non-Gaussian the input currently is. So the per-slice test is Epps–Pulley: project onto
a direction, compare the empirical CF of the projection against the CF of `N(0, 1)` in a Gaussian-weighted
L² norm over a frequency grid, integrate, average over directions, and add to the invariance term.

Verify the statistic does what I need. The CF of `N(0, 1)` is `φ(t) = exp(−t²/2)`, and the Epps–Pulley
weight is the same Gaussian window `exp(−t²/2)` — so target CF and weight coincide (the code's `exp_f`),
and the per-slice statistic is `∫ exp(−t²/2) · |φ̂(t) − exp(−t²/2)|² dt`. Two limits. If the projected
batch is genuinely standard-normal, `φ̂ → exp(−t²/2)`, the integrand → 0, the loss floor sits exactly at
`N(0, 1)`. The collapse limit is the one that has to bite: a projection collapsed to near-constant
`x_b ≈ c` gives `φ̂(t) ≈ e^{itc}`, modulus 1 for every `t`, while the target `exp(−t²/2)` decays away from
`t = 0`; the squared error `1 − 2cos(tc)exp(−t²/2) + exp(−t²)` is zero at `t = 0` but strictly positive
elsewhere, so the weighted integral over `[−3, 3]` is strictly positive — a collapsed slice is a
*high*-loss state. And note what this buys that VICReg needed a separate term for: because I test against
`N(0, 1)` and not merely "Gaussian shape," an under-spread slice fails directly, its CF staying
concentrated near 1 while the standard-normal target has already decayed. The single test enforces the
variance floor, the decorrelation, *and* the higher-moment shape all at once, per view — the anti-collapse
pressure and the distributional target are the same term, the unification the second-moment methods
lacked.

What the harness runs is the BCS (Batched Characteristic Slicing) form, a deliberately stripped version.
I hold `num_slices = 256` unit directions drawn fresh each call from a `torch.Generator` seeded by an
internal `step` counter — so the directions advance and the sphere accumulates coverage — and project
each view, `z @ A`, the whole per-step cost besides the CF being `O(B · D · num_slices)`. The Epps–Pulley
statistic per slice is a mean over the batch at each grid point, `O(B · num_slices · n_points)` — both
linear in B, no `B²` pair term anywhere, which is the concrete payoff of slicing over the multivariate
tests I rejected: the same batch-256 that starves an O(B²) double-sum is plenty for a family of O(B)
per-slice averages. The grid is `t = linspace(−3, 3, 10)`, the complex empirical CF
`ecf = (1j · x_t).exp().mean(0)`, the window `exp(−t²/2)` serving as both target CF and weight, and
`torch.trapz` integrating over `t`. The BCS loss is the mean over the two views' slices; the invariance
term is `F.mse_loss(z1, z2)`; total is `invariance + lmbd · bcs` with `lmbd = 10.0` — the bcs statistic
is a small weighted-integrated quantity while the MSE is O(1), so the 10 lifts the Gaussianity
certificate to a magnitude where it competes with invariance rather than being swamped.

Three things are simpler than the full machinery. It integrates the *full* symmetric grid `[−3, 3]` with
`trapz` rather than folding it onto `[0, 3]` to halve the knots; it keeps complex arithmetic rather than
the algebraically-equal cos/sin-mean real form; and it seeds directions from a plain per-module `step`
counter rather than a cross-device-synchronized one — single device here, no `all_reduce` needed. It also
uses 10 frequency knots, not 17. None change the algorithm; they trade quadrature resolution and
DDP-readiness for a compact single-device module. The config choice that *does* change behaviour is
`CONFIG_OVERRIDES = {"proj_output_dim": 128}`. A Gaussianity-on-random-projections certificate is only as
tight as the density of directions probing the sphere `S^{D−1}`, and covering it to a fixed angular
resolution needs directions growing *exponentially* in D. A fixed budget of 256 slices (plus the
accumulation from resampling) is a far denser cover of `S^{127}` than of `S^{2047}`, so the same
num_slices pins the whole distribution much more tightly at D = 128; fewer output dimensions is also fewer
degrees of freedom to certify Gaussian. This reverses the pressure barlow and VICReg felt — they wanted
width to spread variance across — because SIGReg's anti-collapse term is a distributional certificate that
concentrates as D shrinks. Laid side by side — barlow 2048, VICReg 1024, SIGReg 128 — the projector width
narrows monotonically across the ladder, in step with how much of the distribution each method controls.

So the delta from VICReg is precise: where VICReg pinned only the embedding's second moments per branch —
a variance floor plus a covariance decorrelation, leaving every higher moment free, where the
per-backbone wobble and the ResNet-34 dip lived — SIGReg pins the *whole* per-view embedding distribution
to the derived `N(0, I)` target by slicing it into 256 one-dimensional Epps–Pulley standard-normality
tests in a deliberately narrow 128-dim space, averaged over directions for dense gradients, added to
plain-MSE invariance with weight 10.

Falsifiable expectations against VICReg's 89.85 / 89.5 / 91.38 (and barlow's 89.12 / 90.28 / 90.62).
Because the target distribution is *derived* from the linear- and nonlinear-probe worst case rather than
guessed, and pinning the full distribution controls the higher moments the second-moment methods left
free, I expect low-90s on every backbone, clearing both on aggregate. But the sharper prediction is about
*stability across backbones*: if the wobble — VICReg's ResNet-34 dip below its own ResNet-18, the broken
monotonicity, the spread widening from 1.50 to 1.88 — is the signature of un-pinned higher moments, then
pinning the full distribution should *tighten* the per-backbone spread and restore order, with the
biggest help landing on ResNet-34, VICReg's weakest at 89.5, where second-moment pinning was doing worst.
If instead SIGReg comes in strong on aggregate but the spread stays wide or ResNet-34 stays depressed,
the wobble was *not* about higher moments and my derivation of the target is suspect. And if it lands
merely *level* with VICReg, the likely culprit is the stripped 10-knot full-grid quadrature in the
128-dim space being too coarse to certify Gaussianity tightly — a resolution problem fixable by a finer
grid, not a reason to abandon `N(0, I)`. (The distilled module and the literal scaffold edit are in the
answer.)
