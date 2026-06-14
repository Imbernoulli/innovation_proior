Barlow cleared the floor and it cleared it convincingly: 89.12 on ResNet-18, 90.28 on ResNet-34, 90.62
on ResNet-50, seed 42. So the ~75-point hole the naive MSE left is essentially recovered — the
decorrelation term turned the constant from a global minimizer into a high-loss state and the encoder
now learns class-separable features on all three backbones. Two things in those numbers are worth
reading before I decide what to change. First, the cross-backbone order *did* invert the way I predicted:
under naive collapse ResNet-50 led only on a chance cushion (17.28 vs 14.56), and now with a working
objective the larger backbones genuinely separate classes better — ResNet-50 (90.62) > ResNet-34
(90.28) > ResNet-18 (89.12), a clean monotone with scale, and no straggler stuck near 10%, so the LARS-
starvation failure I was worried about did not bite; `scale_loss = 0.1` tamed the gradient norm enough
for LARS to drive the diagonal to 1. Second, and this is the lever for the next rung, the spread across
backbones is about 1.5 points (89.1 to 90.6) and ResNet-18 in particular sits a little low. That is the
kind of gap I flagged at the close of the barlow derivation: redundancy reduction pins the diagonal to 1
through the *cross*-correlation, which couples the two branches and standardizes the embeddings, and that
coupling-plus-standardization may be shaping a geometry that transfers a hair worse on the smaller
backbone than something cleaner would. So I am not looking for a different *kind* of fix — barlow already
proved the information-maximization family is the right family on this contract — I am looking for a
cleaner member of it.

Let me name exactly what bothers me about barlow's statistic, because the dissatisfaction is the design.
Barlow's whole objective lives in *one* D×D matrix, the cross-correlation between view 1 and view 2. Each
entry `C_ij` mixes a dimension of branch A with a dimension of branch B, so the method implicitly wants
the two branches to have similar output statistics — it *couples* them. And to make the entries genuine
correlations in [−1, 1] so the diagonal target of 1 is meaningful, barlow has to *standardize* the
embeddings (the non-affine BatchNorm), dividing each feature by its batch std. That standardization is a
normalization I am importing into the loss, and the diagonal-to-1 condition is then doing double duty:
it enforces invariance *and*, through the standardization, it is the only thing pinning the per-feature
scale. The redundancy term, meanwhile, only zeros off-diagonals of that same coupled, standardized
matrix. So collapse-prevention in barlow is entangled: it rides on a cross-branch matrix and on an
embedding standardization, and I cannot point at one term and say "this line, by itself, forbids the
constant." I want to *separate* the two failure modes — trivial collapse (everything shrinks to a point)
and informational collapse (the variance crowds into a low-rank subspace) — and forbid each with an
explicit term applied to *each branch on its own*, with no cross-branch matrix and no embedding
standardization. If collapse-prevention is per-branch and local, the two branches need share nothing,
and the geometry is shaped by terms I can read directly rather than by a coupling I have to reason about.

So ask the most literal question. Trivial collapse is exactly "each embedding dimension carries no
variation across the batch — every sample gets the same value." The most direct forbiddance is: look at
one branch's embeddings, compute, per dimension, its variance across the batch, and demand it not be
zero. Force every dimension to vary. That is collapse-prevention stated as bluntly as the failure
itself, with no negatives, no asymmetry, no cross-branch matrix. But I do not want to *maximize*
variance — that would push it without bound and fight the invariance term and just blow the embeddings
up. I want a *floor*: each dimension should have at least some standard deviation, and once above it,
no further pressure. That is a hinge. Define `v(Z) = mean_j max(0, γ − S(z^j))` where `S(z^j)` is the
batch std of column j and γ is the target floor; as long as a dimension's std is ≥ γ it contributes
nothing, and the moment it dips below, the hinge pushes it back up. Apply it to each branch separately —
`v(Z1)` and `v(Z2)` — each from that branch's own batch statistics. No coupling.

A subtlety I almost walked past, and it decides whether the term even works: should the hinge act on the
variance or the standard deviation? They carry the same information, so it feels like it should not
matter. It matters enormously. Put the *variance* in the hinge: `d Var/d z_{b,j} ∝ (z_{b,j} − z̄_j)`, and
near collapse every `z_{b,j}` sits right at the mean, so that gradient ≈ 0 — the term goes silent
exactly in the regime I most need a push *out of*. The variance-in-the-hinge cure does nothing precisely
when the disease is worst. Now the *standard deviation*, `S = √Var`: `dS/dVar = 1/(2√Var) → ∞` as
Var → 0, and `dS/d z_{b,j} = (z_{b,j} − z̄_j)/(B √Var)`; the (z − z̄) factor still shrinks near collapse,
but the deviations are themselves of order √Var, so dividing by √Var keeps the ratio order 1 instead of
vanishing. The restoring force *survives* at collapse. So it has to be the std, with a small ε inside the
root for numerical stability: `S = √(Var + ε)`. With γ = 1 — pick a scale and stick to it, the absolute
value is arbitrary since the network can rescale, what matters is a fixed positive floor — the term is
`mean_j max(0, 1 − √(Var(z^j) + ε))`. Check it kills the constant: a constant gives every column zero
variance, √(0+ε) ≈ 0 < 1, the hinge is fully active on every dimension and pushes them to spread.
Collapse is now the *most* penalized configuration, and I got there with a single per-branch, per-
dimension statistic — no negatives, no asymmetry, no touching the other branch.

But the variance floor is not enough, and the reason is exactly the cheap escape I had to block in
barlow. Suppose every dimension's std is pinned at γ = 1 — floor satisfied — but all D dimensions carry
the *same* signal, copies of one informative direction. Each dimension individually has variance 1, so
the variance term is perfectly happy; the invariance term is happy if that direction is invariant; and
yet the representation has D coordinates but one effective degree of freedom — all the variance crammed
into a tiny subspace, dimensions maximally redundant. That is informational collapse, and the variance
floor is blind to it because variance is a per-dimension statistic that says nothing about whether two
dimensions are the same. So I need a second term forbidding redundancy *between* dimensions — the
decorrelation idea I admired in barlow, but on a single branch, not a cross-correlation between branches.
The natural object is the covariance matrix of one branch's embeddings across the batch,
`C(Z) = (z̃ᵀ z̃)/(B − 1)` after centering. Its diagonal is the per-dimension variances (handled by the
variance term); its off-diagonal `C_ij` (i ≠ j) is how much dimension i and dimension j co-vary —
redundancy is exactly co-variation. Drive every off-diagonal to zero: `c(Z) = Σ_{i≠j} [C(Z)]_{ij}²`
(scaled), computed on each branch separately. And note what I did *not* do: I never normalized the
embeddings into correlations. Barlow had to standardize because its cross-correlation entries needed to
be in [−1, 1] for the diagonal-to-1 target to mean anything. Here I have no diagonal target on the
covariance — the *variance* term already pins each dimension's scale to γ — so I can leave the covariance
unnormalized; the variance term owns the scale and the covariance term decorrelates. That is the
standardization barlow needed and I do not.

The division of labor is the crux, so let me state it. The variance term forbids trivial collapse (each
dimension keeps std ≥ γ, nothing shrinks to a point). The covariance term forbids informational collapse
(decorrelated dimensions, so the guaranteed variance is *spread across all D dimensions* instead of
duplicated). Neither alone suffices: variance alone permits the copy-one-direction redundancy; covariance
alone collapses outright, since the cheapest way to zero all off-diagonals is to send everything to a
constant where the whole covariance matrix is zero. They are complementary — variance gives the
covariance term something to spread, covariance makes the variance meaningful. And invariance is the
third leg, plain MSE between the paired views with no normalization (the variance term owns the scale, so
l2-normalizing would fight it), tying the two views together so the variance-and-decorrelation budget is
spent on augmentation-stable features rather than on independent noise that would satisfy variance and
covariance perfectly while telling me nothing about the image.

Now ground it in *this* harness's edit, because the coefficients are not the generic ones and that is
load-bearing for the numbers. The contract is the same `forward(z1, z2)` → scalar. The invariance term
is `F.mse_loss(z1, z2)` with coefficient 1. The variance term uses `std_margin = 1` and
`std_coeff = 1.0` — the hinge `mean(relu(1 − std))` summed over the two branches, weighted at 1. The
covariance term is the off-diagonal-squared *mean* of each branch's `(xᵀx)/(B−1)` covariance, summed
over the two branches, weighted at `cov_coeff = 100.0`. That cov weight of 100 is far heavier than the
generic recipe's relative weighting, and it pairs with a deliberate reshape: I set
`CONFIG_OVERRIDES = {"proj_output_dim": 1024}`, narrowing the projector output from the default 2048 to
1024. The two choices interlock — a narrower embedding has fewer off-diagonal pairs to decorrelate, so a
large cov coefficient can fully decorrelate the 1024-wide space within budget without the ~D² off-
diagonal gradient destabilizing training, and 1024 is the projector width the upstream "impact of the
projector" comparison ranks best for this method on CIFAR-10 ResNet-18. So unlike barlow, which kept the
default 2048 projector, VICReg here narrows to 1024 and leans hard on covariance (100) with a light
variance floor (1) and unit invariance — the variance term only has to hold the floor while the heavy
covariance term does the spreading, in a deliberately narrower, more fully decorrelated space.

So the delta from barlow is precise: where barlow rode one coupled, standardized cross-correlation
matrix and pinned scale through its diagonal, VICReg splits collapse-prevention into a per-branch
variance floor on the standard deviation (kills trivial collapse, gradient survives at collapse) and a
per-branch unnormalized covariance penalty (kills informational collapse), with plain-MSE invariance,
no embedding standardization and no cross-branch coupling, in a 1024-wide projector with a heavy cov
weight.

Falsifiable expectations against barlow's 89.12 / 90.28 / 90.62. Because this is a cleaner member of the
same family with per-branch terms and a projector width tuned for it, I expect it to land *at or slightly
above* barlow on the larger backbones — high-89s to low-91s — and the interesting bet is ResNet-18,
where barlow sat lowest (89.12): the decoupled per-branch geometry in a 1024-wide space should transfer
at least as well, so I expect VICReg's ResNet-18 to match or edge past barlow's. If the heavy cov weight
of 100 in the narrow 1024 space over-decorrelates before the representation is informative, the risk is
a slightly *lower* ResNet-18 — that would be the tell that the cov weight is too aggressive for the
narrowed projector. The cross-backbone monotone-with-scale that barlow showed should persist (bigger
backbones separate classes better), and I would read a near-tie with barlow on aggregate as confirmation
that on this harness the *kind* of anti-collapse term (decorrelation) matters more than its exact
formulation — which then points the final rung at a regularizer that does not just decorrelate but pins
the *whole* embedding distribution to a target, not only its second moments. (The distilled module and
the literal scaffold edit are in the answer.)
