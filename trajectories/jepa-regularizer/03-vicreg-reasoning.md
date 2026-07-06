Barlow cleared the floor and it cleared it convincingly: 89.12 on ResNet-18, 90.28 on ResNet-34, 90.62
on ResNet-50, seed 42, for an aggregate of (89.12 + 90.28 + 90.62)/3 = 90.01. Set against the naive
floor's 15.20 mean that is a recovery of about 74.8 points — the ~75-point hole the MSE-only objective
left is essentially closed, and it closed on every backbone: ResNet-18 came up +74.56 from its collapsed
14.56, ResNet-34 +76.51 from 13.77, ResNet-50 +73.34 from 17.28. So the decorrelation term did exactly
the structural thing it was supposed to: it turned the constant from a global minimizer into a high-loss
state, and the encoder now learns class-separable features across all three architectures. Two things in
the numbers are worth reading carefully before I decide what to change, because they are the levers for
the next rung.

First, the cross-backbone order inverted from the naive pattern in exactly the way a *working* objective
should force. Under naive collapse the ordering was 17.28 (rn50) > 14.56 (rn18) > 13.77 (rn34) — a
non-monotone jumble, because there the accuracy was collapse-residue noise and ResNet-50 led only on its
wider-feature overfit cushion. Now the ordering is 90.62 (rn50) > 90.28 (rn34) > 89.12 (rn18): a clean
monotone with backbone scale, the larger networks genuinely separating classes better. That flip from
cushion-noise to capacity-signal is itself the confirmation that the objective is doing real
representational work, and that no backbone got stuck — there is no straggler near 10%, so the
LARS-starvation failure I worried about (the raw summed loss being large enough to strangle LARS's
adaptive step) did not bite; `scale_loss = 0.1` tamed the gradient norm enough for LARS to drive the
cross-correlation diagonal to 1 on all three.

Second — and this is the actual lever — the spread across backbones is 90.62 − 89.12 = 1.50 points, and
it is not evenly distributed. The ResNet-34→ResNet-50 step is only 0.34, but the ResNet-18→ResNet-34 step
is 1.16; almost the whole spread is ResNet-18 sitting low. That is the gap I flagged at the close of the
barlow derivation, and its location — the smallest backbone, the one with 512-dim features — is a clue
about mechanism. So I am not looking for a different *kind* of fix; barlow already proved the
information-maximization family is the right family on this contract, recovering the entire collapse hole.
I am looking for a *cleaner member* of it, and specifically one whose geometry does not leave the smallest
backbone lagging.

Let me name exactly what bothers me about barlow's statistic, because the dissatisfaction is the design.
Barlow's whole objective lives in *one* D×D matrix, the cross-correlation between view 1 and view 2, and
at the default projector that is a 2048×2048 object with about 4.19 million entries. Each entry `C_ij`
mixes a dimension of branch A with a dimension of branch B, so the method implicitly wants the two
branches to have matching output statistics — it *couples* them. And to make the entries genuine
correlations in [−1, 1], so that the diagonal target of 1 is even meaningful, barlow has to *standardize*
the embeddings with the non-affine BatchNorm, dividing each feature by its batch std. That standardization
is a normalization I am importing into the loss, and the diagonal-to-1 condition then does double duty: it
enforces invariance *and*, through the standardization, it is the only thing pinning the per-feature
scale. The redundancy term, meanwhile, only zeros the off-diagonals of that same coupled, standardized
matrix. So collapse-prevention in barlow is entangled — it rides on a cross-branch matrix and on an
embedding standardization, and I cannot point at one line and say "this, by itself, forbids the constant."
I want to *separate* the two failure modes — trivial collapse (everything shrinks to a point) and
informational collapse (variance crowds into a low-rank subspace) — and forbid each with an explicit term
applied to *each branch on its own*, with no cross-branch matrix and no embedding standardization. If
collapse-prevention is per-branch and local, the two branches need share nothing, and the geometry is
shaped by terms I can read directly rather than by a coupling I have to reason about — and it is a decent
hypothesis that the coupling-plus-standardization is exactly what costs barlow that point on ResNet-18.

There is more than one way to build a per-branch decorrelation, though, and I should walk the obvious
competitor before I commit, because it looks stronger on paper. The most direct way to decorrelate a
branch's embeddings is to *whiten* them — transform so the batch covariance becomes the identity — which
is what the whitening line of information-maximization SSL does: compute the branch covariance, form its
inverse square root, apply it, then push the whitened views together. Whitening enforces perfect
decorrelation *and* unit variance in one linear operator, which is cleaner than penalizing off-diagonals
softly. But cost it out on this harness. The covariance is D×D and its inverse-square-root is an O(D³)
eigendecomposition or Cholesky every step; at D = 2048 that is 2048³ ≈ 8.6 billion operations per step
just for the whitening matrix, which is heavy but not by itself fatal. What *is* fatal is the rank: the
batch has B = 256 samples and the covariance is 2048×2048, so the empirical covariance has rank at most
255 — it is massively rank-deficient, singular, and its inverse-square-root is ill-posed. Whitening a
2048-dimensional representation from 256 points is trying to estimate 2048 directions of variance from 256
samples; the whitening transform is undefined on the 1793 unseen directions and numerically wild near
them. I could regularize the covariance to make it invertible, but then I am back to a soft penalty with a
tuning knob and I have paid O(D³) for the privilege. So whitening is out at this batch and width. The
alternative keeps the goal — decorrelated, unit-variance-ish per-branch dimensions — but reaches it with
*soft penalties* that never invert the covariance: floor the diagonal, push the off-diagonals toward
zero, each a differentiable term at O(BD²) cost with no rank requirement on the covariance at all. That is
the member of the family I want.

So ask the most literal question. Trivial collapse is exactly "each embedding dimension carries no
variation across the batch — every sample gets the same value." The most direct forbiddance is: look at
one branch's embeddings, compute per dimension its variance across the batch, and demand it not be zero.
Force every dimension to vary. That is collapse-prevention stated as bluntly as the failure itself, with
no negatives, no asymmetry, no cross-branch matrix. But I do not want to *maximize* variance — that would
push it without bound, fight the invariance term, and just blow the embeddings up. I want a *floor*: each
dimension should have at least some standard deviation, and once above it, no further pressure. That is a
hinge, `v(Z) = mean_j max(0, γ − S(z^j))`, with `S(z^j)` the batch std of column j and γ the target floor;
as long as a dimension's std is ≥ γ it contributes nothing, and the moment it dips below, the hinge pushes
it back up. Apply it to each branch separately, `v(Z1)` and `v(Z2)`, each from its own batch statistics.
No coupling.

A subtlety I almost walked past decides whether the term even works: should the hinge act on the variance
or the standard deviation? They carry the same information, so it feels like it should not matter. It
matters enormously, and I can see it in the gradients. Put the *variance* in the hinge:
`∂Var/∂z_{b,j} = (2/B)(z_{b,j} − z̄_j)`, and near collapse every `z_{b,j}` sits right at the mean, so
`(z − z̄) ≈ 0` and that gradient vanishes — the term goes silent exactly in the regime I most need a push
*out of*. The variance-in-the-hinge cure does nothing precisely when the disease is worst. Now the
*standard deviation*, `S = √Var`: by the chain rule `dS/dVar = 1/(2√Var)`, which *diverges* as Var → 0,
and `dS/d z_{b,j} = (z_{b,j} − z̄_j)/(B√Var)`; the `(z − z̄)` factor still shrinks near collapse, but the
deviations are themselves of order √Var, so dividing by √Var keeps the ratio order 1 instead of vanishing.
Put numbers on it: with the ε = 1e-4 floor I use inside the root, at Var = 1e-4 (near collapse)
`dS/dVar = 1/(2·0.01) = 50`, while at Var = 1 (floor satisfied) it is `0.5` — the std-hinge's restoring
gradient is a hundred times stronger near collapse than at the target, self-amplifying exactly where it is
needed. So it has to be the std, with the small ε inside the root for numerical stability:
`S = √(Var + ε)`. With γ = 1 — pick a scale and stick to it, since the absolute value is arbitrary (the
network can rescale) and what matters is a fixed positive floor — the term is
`mean_j max(0, 1 − √(Var(z^j) + ε))`. Check it kills the constant: a constant gives every column zero
variance, `√(0 + 1e-4) = 0.01 < 1`, so `relu(1 − 0.01) = 0.99` fires on *every* dimension, the mean is
≈ 0.99, and summed over the two branches the variance term alone is ≈ 1.98 — collapse is now the *most*
penalized configuration, from a single per-branch, per-dimension statistic, no negatives, no asymmetry, no
touching the other branch.

But the variance floor is not enough, and the reason is exactly the cheap escape I had to block in barlow.
Suppose every dimension's std is pinned at γ = 1 — floor satisfied — but all D dimensions carry the *same*
signal, copies of one informative direction. Each dimension individually has variance 1, so the variance
term is perfectly happy; the invariance term is happy if that direction is invariant; and yet the
representation has D coordinates but one effective degree of freedom — all the variance crammed into a tiny
subspace, dimensions maximally redundant. That is informational collapse, and the variance floor is blind
to it because variance is a per-dimension statistic that says nothing about whether two dimensions are the
same. So I need a second term forbidding redundancy *between* dimensions — the decorrelation idea I admired
in barlow, but on a single branch, not a cross-correlation between branches. The natural object is the
covariance matrix of one branch's embeddings across the batch, `C(Z) = (z̃ᵀ z̃)/(B − 1)` after centering.
Its diagonal is the per-dimension variances (handled by the variance term); its off-diagonal `C_ij`
(i ≠ j) is how much dimension i and dimension j co-vary — and redundancy is exactly co-variation. Drive
every off-diagonal to zero, `c(Z) = Σ_{i≠j} [C(Z)]_{ij}²` (normalized, on each branch separately). And
note what I did *not* do: I never normalized the embeddings into correlations. Barlow had to standardize
because its cross-correlation entries needed to lie in [−1, 1] for the diagonal-to-1 target to mean
anything. Here I have no diagonal target on the covariance — the *variance* term already pins each
dimension's scale to γ — so I can leave the covariance unnormalized; the variance term owns the scale and
the covariance term decorrelates. That is the standardization barlow needed and I do not.

The division of labor is the crux, so let me state it and check that neither term can be dropped. The
variance term forbids trivial collapse (each dimension keeps std ≥ γ, nothing shrinks to a point). The
covariance term forbids informational collapse (decorrelated dimensions, so the guaranteed variance is
*spread across all D dimensions* instead of duplicated). Neither alone suffices, and I can see it in the
corners. Variance alone permits the copy-one-direction redundancy above: every column has std 1 so the
variance term is silent, but the columns are copies, so the covariance off-diagonals are all ≈ 1 and a
covariance term would scream — variance needs covariance to catch that case. Covariance alone collapses
outright: the cheapest way to zero all off-diagonals is to send everything to a constant, where the
*whole* covariance matrix is zero and the off-diagonal penalty is trivially satisfied — pure decorrelation
with no variance floor invites exactly the collapse I am fighting, so covariance needs variance to forbid
that corner. They are complementary: variance gives the covariance term something to spread, covariance
makes the variance meaningful. And invariance is the third leg, plain MSE between the paired views with no
normalization (the variance term owns the scale, so l2-normalizing the embeddings first would fight it),
tying the two views together so the variance-and-decorrelation budget is spent on augmentation-stable
features rather than on independent noise that would satisfy variance and covariance perfectly while
telling me nothing about the image.

Before I trust the three terms together, let me verify the two corners numerically rather than assert
them, and check that the pieces are even comparable in scale. Take the copy-one-direction escape in the
smallest nontrivial case: two dimensions, both unit-variance copies of the same signal. The centered
covariance is then `[[1, 1], [1, 1]]`, its two off-diagonal entries are each 1, so
`_off_diagonal(cov).pow(2).mean() = (1² + 1²)/2 = 1` on that branch — the covariance term reads a full
unit of redundancy and, at `cov_coeff = 100`, returns a large penalty, while the variance hinge sits at
exactly zero because both columns have std 1. That is the case the variance term is blind to, caught
cleanly by the covariance term, and the arithmetic confirms it is a maximal signal, not a marginal one.
Now the joint fixed point the three terms should relax to: invariance wants `z1 = z2` for every image
(zero paired distance), the variance hinge wants each column's std at least 1, and the covariance term
wants the columns decorrelated. All three are simultaneously satisfiable, and essentially in one way —
features that are the *same* across the two augmented views of an image (so invariance is happy) but
*vary* across different images with unit standard deviation and no cross-dimension correlation (so
variance and covariance are happy). That intersection is exactly an augmentation-invariant, unit-scaled,
decorrelated representation — the thing I want — and it is neither the constant (which fails variance),
nor the copy (which fails covariance), nor independent per-view noise (which fails invariance). Finally a
scale check, because three terms only trade off sanely if they are comparable at the target: the
invariance MSE is a mean squared coordinate error, order 1 when embeddings are order 1; the variance
hinge is a difference of standard deviations against γ = 1, order 1; and the covariance `.mean()` is a
mean squared covariance, order 1 when the per-column std is order 1. So near a healthy solution all three
terms live at the same order of magnitude, which is why unit-ish coefficients — once the covariance's
mean-over-D² bookkeeping is accounted for — can balance them without one silently dominating.

Now ground it in *this* harness's edit, because the coefficients are not the generic ones and the
arithmetic behind them is load-bearing for the numbers. The contract is the same `forward(z1, z2)` →
scalar. The invariance term is `F.mse_loss(z1, z2)` with coefficient 1. The variance term uses
`std_margin = 1` and `std_coeff = 1.0` — the hinge `mean(relu(1 − std))` summed over the two branches,
weighted at 1. The covariance term is the off-diagonal-squared *mean* of each branch's `(xᵀx)/(B−1)`
covariance, summed over the two branches, weighted at `cov_coeff = 100.0`. A cov coefficient of 100 looks
alarming next to the canonical VICReg weight of 1, so I want to understand what it actually is before I
trust it. The tell is that `_cov_loss` takes `.pow(2).mean()` over the off-diagonal entries — it divides
the sum of squared off-diagonals by their *count*, `D² − D ≈ D²`, whereas the canonical VICReg covariance
term divides by `D`. So at D = 1024 the code's per-Σ normalization is about `(D − 1) ≈ 1023` times
*smaller* than canonical, and the `cov_coeff = 100` is largely just undoing that: the effective covariance
weight relative to the unit invariance term is `cov_coeff/(D − 1) ≈ 100/1023 ≈ 0.098`, only about two and
a half times the canonical relative cov-to-invariance ratio of `1/25 = 0.04`. So the 100 is not a
genuinely hundred-fold aggressive knob — it is mostly bookkeeping for the mean-over-D² normalization, and
the *effective* decorrelation pressure is moderate, a couple times canonical. That is reassuring, because
a truly 100× covariance weight would blow training up; it also keeps the over-decorrelation risk real but
*modest* rather than catastrophic.

The cov weight pairs with a deliberate reshape: `CONFIG_OVERRIDES = {"proj_output_dim": 1024}`, narrowing
the projector output from the default 2048 to 1024. The two choices interlock through the number of
constraints. The covariance term asks every off-diagonal pair to hit zero simultaneously, and there are
`D(D − 1)/2` such pairs — about 523k at D = 1024 versus about 2.1M at D = 2048, a 4× reduction. Fewer
simultaneous zero-constraints means a fixed optimization budget can decorrelate the narrower space more
*fully*, so the guaranteed variance really does end up spread across all 1024 dimensions rather than
leaving a residue of correlated pairs. And 1024 is the projector width the upstream "impact of the
projector" comparison ranks best for VICReg on CIFAR-10 ResNet-18, which is the corroborating evidence
that this is the right width for *this* method. So unlike barlow, which kept the default 2048 projector,
VICReg here narrows to 1024 and applies an effectively-moderate covariance pressure (nominal 100, effective
≈ 0.1 relative) with a light variance floor (1) and unit invariance — the variance term holds the floor
while the covariance term does the spreading, in a deliberately narrower, more fully decorrelated space.

So the delta from barlow is precise: where barlow rode one coupled, standardized cross-correlation matrix
and pinned scale through its diagonal, VICReg splits collapse-prevention into a per-branch variance floor
on the standard deviation (kills trivial collapse, gradient survives at collapse) and a per-branch
unnormalized covariance penalty (kills informational collapse), with plain-MSE invariance, no embedding
standardization and no cross-branch coupling, in a 1024-wide projector with an effectively-moderate cov
weight.

Falsifiable expectations against barlow's 89.12 / 90.28 / 90.62. Because this is a cleaner member of the
same family with per-branch terms and a projector width tuned for it, I expect it to land *at or slightly
above* barlow on the larger backbones — high-89s to low-91s. The interesting bet is ResNet-18, where
barlow sat lowest at 89.12 and where I argued the coupling-plus-standardization was costing the most: the
decoupled per-branch geometry in the narrowed 1024-wide space should transfer at least as well, so I
expect VICReg's ResNet-18 to match or edge past barlow's. If instead the covariance pressure — moderate
though I computed it to be — over-decorrelates in the narrow space before the representation is fully
informative, the tell would be a ResNet-18 that comes in *below* barlow's, and I would read that as the
cov weight being a touch too aggressive for the narrowed projector. The monotone-with-scale ordering that
barlow showed should persist, since bigger backbones separate classes better once the objective is
working. And here is the outcome I am half-expecting, and what I would make of it: a *near-tie* with barlow
on aggregate. If two quite different second-moment formulations — barlow's coupled cross-correlation and
VICReg's per-branch variance-plus-covariance — land within a fraction of a point of each other, that says
the *kind* of anti-collapse term (some form of decorrelation) matters more on this harness than its exact
formulation, and that both are bumping against the same ceiling. That would point the final rung somewhere
neither of these two reaches: a regularizer that does not merely decorrelate or floor variance — that does
not merely pin the embedding's *second moments* — but pins the *whole* embedding distribution to a target,
controlling the higher moments both of these leave free. (The distilled module and the literal scaffold
edit are in the answer.)
