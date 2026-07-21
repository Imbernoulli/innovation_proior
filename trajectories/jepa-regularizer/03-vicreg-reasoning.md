Barlow cleared the floor convincingly: 89.12 on ResNet-18, 90.28 on ResNet-34, 90.62 on ResNet-50,
aggregate (89.12 + 90.28 + 90.62)/3 = 90.01 against the naive floor's 15.20 — a recovery of about 74.8
points, and it closed on every backbone (+74.56 from 14.56, +76.51 from 13.77, +73.34 from 17.28). The
decorrelation term did exactly the structural thing it was supposed to: it turned the constant from a
global minimizer into a high-loss state, and the encoder now learns class-separable features across all
three architectures. Two things in the numbers are the levers for the next step.

First, the cross-backbone order inverted from the naive scramble in exactly the way a *working*
objective should. Under collapse it was 17.28 (rn50) > 14.56 (rn18) > 13.77 (rn34) — non-monotone,
because accuracy was residue-noise and ResNet-50 led only on its wider-feature cushion. Now it is 90.62
> 90.28 > 89.12: a clean monotone with scale, the larger networks genuinely separating classes better,
no straggler near 10%. So the LARS-starvation failure I worried about did not bite; `scale_loss = 0.1`
tamed the gradient norm enough for LARS to drive the cross-correlation diagonal to 1 on all three.
Second — the actual lever — the spread is 90.62 − 89.12 = 1.50 points and it is not evenly distributed:
the rn34→rn50 step is only 0.34, but rn18→rn34 is 1.16. Almost the whole spread is ResNet-18 sitting
low, and its location — the smallest backbone, 512-dim features — is a clue about mechanism. So I am not
looking for a different *kind* of fix; barlow proved the information-maximization family is right on this
contract. I am looking for a *cleaner member* of it, one whose geometry does not leave the smallest
backbone lagging.

Name exactly what bothers me about barlow's statistic, because the dissatisfaction is the design.
Barlow's objective lives in *one* D×D cross-correlation matrix between the two views — at the default
projector a 2048×2048 object — and each entry mixes a dimension of branch A with a dimension of branch B,
so the method *couples* the branches. And to make the entries genuine correlations in `[−1, 1]`, so the
diagonal-to-1 target means anything, it must *standardize* the embeddings with the non-affine BatchNorm.
That standardization does double duty — it enforces invariance and it is the only thing pinning the
per-feature scale — while the redundancy term only zeros off-diagonals of that same coupled,
standardized matrix. So collapse-prevention is entangled across a cross-branch matrix and an embedding
standardization; I cannot point at one line and say "this, by itself, forbids the constant." I want to
*separate* the two failure modes — trivial collapse (everything shrinks to a point) and informational
collapse (variance crowds into a low-rank subspace) — and forbid each with an explicit term applied to
*each branch on its own*, no cross-branch matrix, no embedding standardization. It is a decent hypothesis
that the coupling-plus-standardization is what costs barlow that point on the smallest backbone.

There is more than one way to build a per-branch decorrelation, and the obvious competitor looks stronger
on paper: *whiten* each branch — transform so the batch covariance becomes the identity — which is what
the whitening line of information-maximization SSL does. Whitening enforces perfect decorrelation *and*
unit variance in one linear operator. But cost it out on this harness. The covariance is D×D and its
inverse-square-root is an O(D³) eigendecomposition every step; at D = 2048 that is ~8.6 billion ops per
step, heavy but not by itself fatal. What *is* fatal is the rank: with B = 256 the empirical covariance
has rank at most 255, so a 2048×2048 covariance is massively rank-deficient and its inverse-square-root
is ill-posed — undefined on the ~1793 unseen directions and numerically wild near them. I could
regularize the covariance to invert it, but then I am back to a soft penalty with a tuning knob and have
paid O(D³) for the privilege. So whitening is out at this batch and width. The alternative keeps the goal
— decorrelated, unit-variance-ish per-branch dimensions — but reaches it with *soft penalties* that never
invert the covariance, at O(BD²) cost with no rank requirement.

So ask the most literal question. Trivial collapse is exactly "each embedding dimension carries no
variation across the batch — every sample gets the same value." The most direct forbiddance: compute each
dimension's variance across the batch and demand it not be zero. But I do not want to *maximize* variance
— that would push it without bound and fight the invariance term — I want a *floor*: each dimension at
least some standard deviation, and no further pressure above it. That is a hinge,
`v(Z) = mean_j max(0, γ − S(z^j))`, with `S(z^j)` the batch std of column j and γ the floor. Apply it to
each branch separately, `v(Z1)` and `v(Z2)`, each from its own statistics. No coupling.

A subtlety decides whether the term even works: should the hinge act on the variance or the standard
deviation? They carry the same information, but the gradients differ enormously. With the *variance*,
`∂Var/∂z_{b,j} = (2/B)(z_{b,j} − z̄_j)`, and near collapse every value sits at the mean, so `(z − z̄) ≈ 0`
and the gradient vanishes — the term goes silent exactly in the regime I most need a push *out of*. With
the *standard deviation* `S = √Var`, `dS/dVar = 1/(2√Var)` *diverges* as Var → 0: the `(z − z̄)` factor
still shrinks near collapse, but the deviations are themselves order √Var, so dividing by √Var keeps the
restoring gradient order 1 instead of vanishing. With the ε = 1e-4 floor inside the root,
`dS/dVar = 1/(2·0.01) = 50` at Var = 1e-4 versus `0.5` at Var = 1 — a hundred times stronger near
collapse than at the target, self-amplifying where it is needed. So it must be the std, `S = √(Var + ε)`.
With γ = 1 — the absolute scale is arbitrary since the network can rescale, what matters is a fixed
positive floor — the term is `mean_j max(0, 1 − √(Var(z^j) + ε))`. Check the constant: every column has
zero variance, `√(1e-4) = 0.01 < 1`, so `relu(1 − 0.01) = 0.99` fires on every dimension, and summed
over the two branches the variance term alone is ≈ 1.98 — collapse is now the most-penalized
configuration, from a single per-branch, per-dimension statistic.

But the variance floor is blind to the cheap escape I had to block in barlow. Pin every std at γ = 1 yet
make all D dimensions copies of one informative direction: each has variance 1 so the variance term is
happy, invariance is happy if that direction is invariant, and yet the representation has D coordinates
but one effective degree of freedom — informational collapse. Variance is a per-dimension statistic and
says nothing about whether two dimensions are the same. So I need a second term forbidding redundancy
*between* dimensions — barlow's decorrelation idea, but on a single branch, not a cross-correlation. The
object is the covariance of one branch across the batch, `C(Z) = (z̃ᵀ z̃)/(B − 1)` after centering; its
off-diagonal `C_ij` is how much dimensions i and j co-vary, and redundancy *is* co-variation. Drive every
off-diagonal to zero, `c(Z) = Σ_{i≠j} [C(Z)]_{ij}²`, per branch. And note what I did *not* do: I never
normalized the embeddings into correlations. Barlow had to standardize so its cross-correlation entries
lay in [−1, 1] for the diagonal target; here I have no diagonal target on the covariance — the *variance*
term already pins each dimension's scale — so I leave the covariance unnormalized. The variance term owns
the scale, the covariance term decorrelates. That is the standardization barlow needed and I do not.

Neither term can be dropped, and the corners show why. Variance alone permits the copy-one-direction
escape: take two unit-variance copies of one signal, the centered covariance is `[[1, 1], [1, 1]]`, its
two off-diagonals are each 1, so `_off_diagonal(cov).pow(2).mean() = (1 + 1)/2 = 1` reads a full unit of
redundancy and — at `cov_coeff = 100` — fires hard, while the variance hinge sits at exactly zero.
Covariance alone collapses outright: the cheapest way to zero all off-diagonals is to send everything to
a constant, where the whole covariance is zero. So variance needs covariance to catch the copy,
covariance needs variance to forbid the constant — complementary. Invariance is the third leg, plain MSE
with no normalization (the variance term owns the scale, so l2-normalizing first would fight it), tying
the two views together so the variance-and-decorrelation budget is spent on augmentation-stable features
rather than on independent per-view noise that would satisfy both while telling me nothing about the
image. The joint fixed point the three relax to is essentially unique: features *same* across the two
views (invariance), *varying* across images at unit std (variance), *decorrelated* (covariance) — an
augmentation-invariant, unit-scaled, decorrelated representation, which is neither the constant nor the
copy nor per-view noise. All three terms live at order 1 near that solution, so unit-ish coefficients
balance them once the covariance's mean-over-D² bookkeeping is accounted for.

That bookkeeping is the crux of the coefficients, and `cov_coeff = 100` looks alarming next to the
canonical VICReg weight of 1 until I see what it is. `_cov_loss` takes `.pow(2).mean()` over the
off-diagonal entries — dividing the sum of squared off-diagonals by their count `D² − D ≈ D²` — whereas
canonical VICReg divides by `D`. So at D = 1024 the code's per-matrix normalization is about `(D − 1) ≈
1023` times smaller than canonical, and the 100 is largely undoing that: the effective covariance weight
relative to the unit invariance term is `100/1023 ≈ 0.098`, only about two and a half times the canonical
relative cov-to-invariance ratio of `1/25 = 0.04`. So the 100 is not a genuinely hundred-fold aggressive
knob — it is mostly bookkeeping for the mean-over-D² normalization, and the *effective* decorrelation
pressure is moderate. That keeps the over-decorrelation risk real but *modest* rather than catastrophic.

The cov weight pairs with a deliberate reshape, `CONFIG_OVERRIDES = {"proj_output_dim": 1024}`, narrowing
the projector output from 2048 to 1024. The two interlock through the number of constraints: the
covariance term asks every off-diagonal pair to hit zero simultaneously, and there are `D(D − 1)/2` such
pairs — about 523k at D = 1024 versus 2.1M at D = 2048, a 4× reduction. Fewer simultaneous
zero-constraints means a fixed budget can decorrelate the narrower space more *fully*, so the guaranteed
variance really spreads across all 1024 dimensions rather than leaving a residue of correlated pairs. And
1024 is the projector width the upstream projector comparison ranks best for VICReg on CIFAR-10
ResNet-18. So unlike barlow's default 2048, VICReg narrows to 1024 with an effectively-moderate covariance
pressure and a light variance floor — the variance term holds the floor while the covariance does the
spreading, in a deliberately narrower, more fully decorrelated space.

So the delta from barlow is precise: where barlow rode one coupled, standardized cross-correlation matrix
and pinned scale through its diagonal, VICReg splits collapse-prevention into a per-branch variance floor
on the standard deviation (kills trivial collapse, gradient survives at collapse) and a per-branch
unnormalized covariance penalty (kills informational collapse), with plain-MSE invariance, no embedding
standardization and no cross-branch coupling, in a 1024-wide projector.

Falsifiable expectations against barlow's 89.12 / 90.28 / 90.62. As a cleaner member of the same family
with per-branch terms and a width tuned for it, I expect it to land *at or slightly above* barlow on the
larger backbones — high-89s to low-91s — and the monotone-with-scale to persist. The interesting bet is
ResNet-18, barlow's weakest at 89.12, exactly where I argued the coupling-plus-standardization costs
most: the decoupled per-branch geometry in the narrowed 1024-wide space should transfer at least as well,
so I expect VICReg's ResNet-18 to match or edge past. A ResNet-18 that comes in *below* barlow's would
mean the covariance pressure — moderate though I computed it — over-decorrelates in the narrow space
before the representation is fully informative. And the outcome I am half-expecting is a *near-tie* on
aggregate: if two quite different second-moment formulations — barlow's coupled cross-correlation and
VICReg's per-branch variance-plus-covariance — land within a fraction of a point, that says the *kind* of
anti-collapse term matters more on this harness than its exact formulation, and both are bumping the same
ceiling. That would point the final step somewhere neither reaches: pinning not the embedding's *second*
moments but its *whole* distribution to a target, controlling the higher moments both of these leave
free. (The distilled module and the literal scaffold edit are in the answer.)
