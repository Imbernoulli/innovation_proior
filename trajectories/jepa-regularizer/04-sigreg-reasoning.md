VICReg landed where I bet it would and the numbers settle the family question. Seed 42 gave 89.85 on
ResNet-18, 89.5 on ResNet-34, 91.38 on ResNet-50 — aggregate 90.24 against barlow's 90.01, a near-tie.
Read the two side by side and the diagnosis is sharp. On ResNet-18 VICReg edged barlow (89.85 vs 89.12),
exactly the bet I made: the decoupled per-branch geometry in the narrowed 1024-wide projector transfers
a touch better on the smallest backbone. On ResNet-50 it pulled clearly ahead (91.38 vs 90.62). But on
ResNet-34 it actually slipped below barlow (89.5 vs 90.28) and below its own ResNet-18 — the monotone-
with-scale that both barlow and the naive cushion showed is broken here, with the middle backbone the
odd one out. That non-monotonicity is the tell. The heavy `cov_coeff = 100` in the narrow 1024 space is
doing real work on ResNet-18 and ResNet-50 but is evidently a little aggressive on ResNet-34 — over-
decorrelating before the representation is fully informative on that one architecture, exactly the risk I
flagged. So the deeper lesson from the barlow→VICReg step is not "VICReg beats barlow"; it is that two
different second-moment formulations land within ~0.2 of each other on aggregate while *neither* is stable
across all three backbones. Both pin only the embedding's *second moments* — barlow zeros a cross-
correlation, VICReg floors per-dimension variance and zeros per-branch covariance — and a method that
controls only the first two moments leaves the rest of the distribution free, which is precisely where
the per-backbone wobble lives. The opening I named at the close of the VICReg derivation is now forced:
stop pinning second moments and pin the *whole* embedding distribution to a target.

So go back to the question both barlow and VICReg answered only implicitly: what distribution *should*
the embeddings have? Neither method derived a target; each picked a statistic (decorrelation, variance
floor) and hoped the resulting geometry transfers. Let me actually derive the target, because if I know
the right distribution I can regularize *toward it* directly and stop guessing at coefficients. The honest
constraint is that I do not know the downstream task at pretraining time — the whole point is a frozen
representation probed for tasks I have not seen — so I should pick the embedding geometry best in the
*worst case* over unknown tasks. Take the simplest probe, the linear one this harness actually uses: fit
ridge regression to unknown labels on the frozen features. The ridge bias is
`−λ(ZᵀZ + λI)⁻¹β_true`, and diagonalizing the Gram matrix, the bias along the *weakest* eigendirection is
`λ/(λ_min + λ)` versus `λ/(λ̄ + λ)` for an isotropic spectrum of the same total energy; since λ_min < λ̄
whenever the spectrum is not flat, anisotropy strictly hurts some downstream task. And the unregularized
variance is `σ² Σ_j 1/λ_j`, which by Jensen (1/x convex) is minimized at fixed mean eigenvalue exactly
when all eigenvalues are equal. So the linear probe — the very metric on the leaderboard — says the
embeddings should be **isotropic**. That already explains the per-backbone wobble: barlow and VICReg push
toward isotropy through second moments but do not pin it, so each backbone lands at a different anisotropy.

Isotropy fixes only the covariance shape, though, and a worst-case-over-tasks argument should care about
more than two moments — which is the whole reason second-moment methods wobble. Push to a nonlinear probe,
a radius-kNN or kernel estimator. Its leading integrated squared bias depends on the Fisher-information
functional `J(p) = ∫ ‖∇log p‖² p dx` of the embedding density. Which density minimizes Fisher information
at fixed covariance? The identity estimator `T(X) = X` is unbiased for the location of `p_θ(x) = p(x−θ)`
with covariance Σ, so the Cramér–Rao bound gives Σ ⪰ I(θ)⁻¹, hence I ⪰ Σ⁻¹, hence J(p) = tr I ≥
tr(Σ⁻¹), with equality iff the score is affine in x — i.e. iff `p` is **Gaussian**. So among all densities
with a given covariance, the Gaussian uniquely minimizes the functional the nonlinear-probe bias depends
on. Combine with isotropy: minimizing tr(Σ⁻¹) = Σ_i 1/λ_i under any scalar covariance constraint forces
Σ = sI. The two axioms drop out together — the embeddings should be distributed as an **isotropic
Gaussian** `N(0, I)`. That is a target neither barlow nor VICReg has; they enforce *consequences* of it
(decorrelation is "off-diagonals of Σ are zero," the variance floor is "diagonal of Σ is bounded below")
but never the distribution itself. Pinning `N(0, I)` subsumes both — isotropic covariance is decorrelated
and equal-variance — and additionally controls every higher moment, which is the part the second-moment
methods leave free and where the per-backbone wobble lives.

Now the hard part: how do I push a batch of embeddings toward `N(0, I)` cheaply? The naive instinct is a
divergence — KL or MMD — between the empirical embedding distribution and the target, but estimating any
of those in D-dimensional space is the high-dimensional density-estimation problem I am trying to avoid.
Reframe it as a hypothesis test: is there evidence the embedding distribution differs from `N(0, I)`?
Pick a scalar test statistic that measures departure and *minimize* it as the regularizer, driving the
encoder toward configurations with no detectable non-Gaussianity. The off-the-shelf multivariate
normality tests (Baringhaus–Henze, Henze–Zirkler) are double sums over all pairs of samples — O(B²),
coupling every sample to every other. That is exactly the sample-hungry, batch-coupled cost that hobbled
contrastive learning and that this batch-256 harness punishes. The escape is *slicing*. Cramér and Wold:
two random vectors are equal in distribution iff all their one-dimensional projections are. So instead of
testing the D-dimensional law, test the univariate projections `aᵀz` against the projection of the target
— and for `N(0, I)`, every unit projection `aᵀz` is exactly `N(0, 1)`, so each univariate test is just
"are these projected scalars standard normal?" A D-dimensional Gaussianity test becomes a family of one-
dimensional ones, one per random direction, each operating on a scalar batch. Sample a finite set of unit
directions and aggregate. For a *loss* I aggregate by *averaging* over directions rather than taking the
max — the max routes gradient only through the single worst direction (sparse, jumpy), while the average
routes gradient through all sampled directions at once, which is what stable optimization needs. The
remaining choice is which univariate goodness-of-fit statistic to put on each slice. Moment-based tests
(skewness/kurtosis) are non-identifiable for finite order and explode in gradient for high order; CDF-
based tests (Cramér–von Mises, Anderson–Darling, KS) need sorting, which is non-differentiable. The
characteristic-function family threads the needle: the empirical characteristic function
`φ̂(t) = mean_b exp(i t x_b)` is a differentiable *average*, identifiable (it is the full Fourier content,
not a truncation), and — for the Epps–Pulley weighted-L² distance to the target CF — has bounded gradient
and curvature regardless of the input distribution. So the per-slice test is Epps–Pulley: project onto a
direction, compare the empirical CF of the projection to `exp(−t²/2)` (the CF of `N(0,1)`) in a Gaussian-
weighted L² norm, integrate over a frequency grid, average over directions, and add it to the invariance
(predictability) term with a single coefficient.

Now ground it in *this* harness's edit, because what it runs is the BCS (Batched Characteristic Slicing)
form, a deliberately stripped version of the full machinery, and the simplifications are load-bearing for
both the cost and the numbers. The contract is the same `forward(z1, z2)` → scalar. I hold `num_slices =
256` random directions, drawn fresh each call from a `torch.Generator` seeded by an internal `step`
counter (so the directions advance over training and the sphere accumulates coverage), normalized to unit
norm. I project each view onto them. The Epps–Pulley statistic per projection is computed directly: a
small frequency grid `t = linspace(−3, 3, 10)`, the complex empirical CF `ecf = mean_b exp(i·t·xᵀ)` via
`(1j * x_t).exp().mean(0)`, the Gaussian window `exp(−t²/2)` serving as both target CF and weight, the
weighted squared error `exp(−t²/2)·|ecf − exp(−t²/2)|²`, and `torch.trapz` over t. The BCS loss is the
mean of that over the two views' slices; the invariance term is `F.mse_loss(z1, z2)`; total is
`invariance + lmbd · bcs` with `lmbd = 10.0`. Three things are *simpler* than the full LeJEPA-style
machinery and I should be explicit that the harness omits them: it integrates the *full* symmetric grid
[−3, 3] with `torch.trapz` rather than exploiting evenness to halve it; it keeps complex arithmetic
(`1j`) rather than the cos/sin-mean real form; and it seeds directions from a plain per-module `step`
rather than a cross-device-synchronized counter with an `all_reduce` (single-device here, so no sync is
needed). It also uses only 10 frequency knots, not 17. None of these change the algorithm — they trade a
little quadrature resolution and DDP-readiness for a compact single-device implementation. The one
configuration choice that *does* matter: `CONFIG_OVERRIDES = {"proj_output_dim": 128}`, narrowing the
projector output to 128. The Gaussianity-on-random-projections test concentrates better at low output
dim — fewer directions are needed to cover a 128-sphere than a 2048-sphere, so 256 slices pin the whole
distribution far more tightly at D = 128, and the upstream sigreg recipe ranks 2048→128 top. So unlike
barlow (2048) and VICReg (1024), SIGReg narrows hardest, to 128, because its anti-collapse term *wants* a
low-dimensional space to certify Gaussian.

So the delta from VICReg is precise: where VICReg pinned only the embedding's second moments per branch —
variance floor plus covariance decorrelation, the part that left every higher moment free and wobbled per
backbone — SIGReg pins the *whole* per-view embedding distribution to the derived `N(0, I)` target by
slicing it into 256 one-dimensional Epps–Pulley Gaussianity tests in a deliberately narrow 128-dim space,
averaged for dense gradients, added to plain-MSE invariance.

Falsifiable expectations against VICReg's 89.85 / 89.5 / 91.38 (and barlow's 89.12 / 90.28 / 90.62).
Because the target distribution is *derived* from the linear- and nonlinear-probe worst case rather than
guessed, and because pinning the full distribution controls the higher moments the second-moment methods
left free, I expect SIGReg to clear both on aggregate — low-90s on every backbone — and, more tellingly,
to be the most *stable across backbones*: I expect the per-backbone spread to tighten (no ResNet-34-style
dip below its own ResNet-18), because the wobble was the signature of un-pinned higher moments. The clean
prediction is every backbone at or above ~90.5 with the three within roughly half a point of each other,
beating VICReg's ResNet-34 (89.5) by the largest margin precisely where second-moment pinning was
weakest. The risk is that the stripped 10-knot full-grid quadrature in the 128-dim space is too coarse to
certify Gaussianity tightly — if so I would see it land merely *level* with VICReg rather than above, and
the diagnosis would be quadrature resolution, not the target. (The distilled module and the literal
scaffold edit are in the answer.)
