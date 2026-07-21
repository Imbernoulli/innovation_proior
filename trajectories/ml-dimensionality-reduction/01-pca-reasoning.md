The embedding is the whole point, but it bolts onto a fixed scoring harness, and the floor I start from
is the cheapest faithful projection I can write into the scaffold's `fit_transform`. The scaffold
default hands me a random linear projection: orthonormalize a couple of Gaussian axes and project. That
is the Johnson–Lindenstrauss construction, and at a target of two dimensions it is a genuinely poor
floor — JL only promises distance preservation when the target dimension is large enough to host the
random directions, and two is nowhere near that, so a random pair of axes captures whatever arbitrary
slice of the variance it happens to land on, usually a thin one. So the first real question is the right
*first rung*: the cheapest projection that is not arbitrary — one that picks its directions from the
data instead of from a random number generator — and what to expect it to do and to fail at on these
three datasets, because that failure is what forces the second rung.

I can make "genuinely poor" a number. The lemma says holding every pairwise distance among n points
inside a factor 1 ± ε needs a target dimension of order ln(n)/ε^2; writing it as m ≥ 8 ln(n)/ε^2 with
n = 5000, ln(5000) ≈ 8.52, even a loose ε = 0.5 demands m ≥ 8·8.52/0.25 ≈ 273 dimensions. Run it
backwards at the m = 2 I am actually handed: 2 = 8·8.52/ε^2 forces ε ≈ 5.8, a "distortion" far larger
than one, which is the bound's way of saying it guarantees nothing at all. So the random projection is
not a little worse than a data-driven map, it is off the edge of the theorem that justifies it, and the
shape of the fix follows: at two dimensions I have no room to be indifferent about *which* two
directions I keep, so I must spend them on the directions the data itself says are richest.

The prior art names three linear candidates for "directions from the data": a least-squares regression
line, factor analysis with a fixed number of latent factors, and the variance-ranking idea. Two fall to
concrete objections. Take the regression line: pick one coordinate as the thing to predict and minimize
the sum of squared residuals measured *along that coordinate's axis*. Swap which coordinate I call
dependent and minimize residuals along the *other* axis, on the very same points, and I get a different
line — two lines for one cloud. On standardized data with unit variances and correlation ρ, regressing
the second coordinate on the first gives slope ρ, the reverse gives slope 1/ρ; at ρ = 0.8 that is slope
0.8 against slope 1.25, coinciding only in the degenerate ρ = 1. The reason is structural: a residual
measured along a single coordinate secretly assumes that coordinate is exact and the other carries all
the error. On standardized pixel data that is false — every coordinate is a measurement with its own
wobble, none is privileged. I need a fit symmetric in all the coordinates that returns one unique flat.

Factor analysis is the subtler candidate, worth ruling out by what it costs. It models each observed
coordinate as a linear combination of a few latent factors plus per-coordinate noise, so the covariance
is C ≈ Λ Λ^T + Ψ with Λ the loadings and Ψ a diagonal of private noise variances. Set Ψ to a single
isotropic level, Ψ = τ^2 I, and this collapses to probabilistic PCA whose factor directions *are* the
top eigenvectors of C — so the only thing factor analysis adds over the variance idea is a
*heteroscedastic* Ψ, a separate estimated noise level per pixel. That buys nothing I need for a 2D
projection and costs a great deal: on the 784-pixel images Ψ carries 784 extra parameters with no
closed form, fit by an EM loop, and the whole apparatus exists to *down-weight* noisy coordinates rather
than to find a faithful low-dimensional frame. Under a five-minute CPU budget, paying for an iterative
noise model to recover at best a reweighting of the same eigen-directions is a bad trade. So the survivor
is the third candidate, built from the symmetric-fit requirement the regression line failed.

A symmetric notion of "distance from a point to a line" is not the vertical or horizontal gap but the
*perpendicular* distance — it treats every direction alike, not caring which axis I call dependent. Make
the criterion the sum of squared perpendicular distances p_k and minimize U = sum of p_k^2. Rotate the
coordinate frame however I like and U does not change, because perpendicular distance is a geometric
quantity. That is the right objective; solving it is what tells me why this rung behaves the way it does.

One recognition saves a lot of grinding. U is the second moment of the point system about the line, and
the second moment about a family of *parallel* axes is least for the axis through the centroid — so the
best-fitting line must pass through the mean. It falls out of the algebra: write the flat as l·x = p
with ||l|| = 1, the perpendicular distance of a point is l·x − p, and differentiating U with respect to
p alone gives −2·sum(l·x − p) = 0, i.e. p = l·xbar. So the first concrete instruction the rung gives me
is **center the data**, then find directions. From here I shift the origin to the centroid and hunt only
for a *direction*.

Bring in the unit-length constraint with a Lagrange multiplier and vary the direction: after
substituting p = l·xbar the centered objective is U = sum of (l·(x − xbar))^2, and stationarity gives,
component by component, C l = σ^2 l — l is an eigenvector of the covariance matrix and the eigenvalue is
the residual variance left perpendicular to the flat. For a plane I want that residual minimal, so I
take the *least* eigenvalue and its eigenvector as the discarded normal.

Run the eigenproblem on a case I can solve by hand. Two standardized coordinates with correlation 0.8,
C = [[1, 0.8], [0.8, 1]]: (1 − λ)^2 = 0.64 gives λ = 1.8 and 0.2, with eigenvectors (1, 1)/√2 and
(1, −1)/√2. The dominant direction is the 45° diagonal — and the detail that convinces me the object is
right is that it sits *between* the two regression lines of slope 0.8 and 1.25, bisecting them, exactly
what a symmetric fit should do where the two asymmetric fits disagree. There is a clean degenerate case
too: an isotropic ball — all correlations zero, all variances equal — has no preferred axis, and the
method correctly refuses to pick a direction when the data has none.

But I care about the best-fitting *line* — the dominant direction of the cloud — more than the discarded
normal. The squared perpendicular distance of a centered point from the line with unit direction l is,
by Pythagoras, the full squared length minus the squared projection onto l. Summed and divided by n, the
perpendicular residual equals (total per-coordinate variance) minus (variance projected onto l). The
first term is a *constant* independent of l; the second is the variance of the data *along* the line. So
minimizing the perpendicular residual is identical to *maximizing the variance of the projection* —
total scatter splits Pythagoreanly into "along" plus "perpendicular" and the total is fixed. And the
line that maximizes projected variance is the eigenvector of C with the *greatest* eigenvalue. The
best-fitting line is the direction of greatest variance; the best-fitting plane is perpendicular to the
direction of least variance.

I want an *ordered sequence* of new coordinates, not just one. Ask for mutually-uncorrelated
unit-variance variables such that each observed coordinate is a linear combination of them, the first
accounting for as much of the total variance as possible, the second for as much of what is left. If the
first component is a unit-variance score w·z, its loading is a = C w and the variance it accounts for is
w^T C^2 w; maximizing subject to w^T C w = 1 collapses again to C a = k a, the loading the eigenvector,
k the variance accounted for. So the two routes — minimize perpendicular residual and maximize explained
variance — land on the *same* eigenproblem of the second-moment matrix, and det(C − kI) = 0 is the
characteristic equation Cauchy studied for the principal axes of quadrics; its roots are real (symmetric
matrix) and here non-negative, because each root is a variance. Real, non-negative, ordered — exactly
what an importance ranking needs. Successive components write themselves: the next explains as much of
the *residual* variance as possible, the top eigenvector of the deflated matrix, and eigenvector
orthogonality makes this automatic, so the full eigendecomposition hands me every component at once,
ordered by eigenvalue. For a 2D embedding I keep the top two.

Now *how* to compute the directions, because the scaffold's budget is real and the numerics matter.
Forming C = Xc^T Xc squares the condition number of Xc: small singular values get squared and drown in
roundoff, so I lose accuracy precisely on the small-variance directions. The numerically sound move is
the SVD of the centered data directly, Xc = U S V^T. Then Xc^T Xc = V S^2 V^T, so the right singular
vectors V are the eigenvectors of the covariance — my principal directions — the variance of component i
is S_i^2/(n−1), and singular values come sorted so I need not sort eigenvalues. Better, the embedding is
the top columns of U S since Xc V = U S: I read the coordinates straight off the factorization. One
reproducibility wrinkle the harness cares about: singular-vector signs are arbitrary
(U S V^T = (−U) S (−V)^T), so a deterministic sign rule — force each vector's largest-magnitude entry
positive — keeps the output stable across the seeds.

The linear-algebra core is nowhere near the five-minute wall: a full SVD of a 5000×784 matrix is
O(n p^2) ≈ 3×10^9 flops, a few seconds, and a randomized SVD keeping only the top two directions is
O(n p k) ≈ 10^7, essentially instantaneous — which is why the scaffold can thread `random_state` through
a randomized solver without noticing the cost. U S truncated to two columns is (n, 2), exactly the shape
the harness asserts. The cost of the whole ladder will live in the *nonlinear* rungs, not here.

So the first rung: center the standardized cloud, take the top two right singular vectors of the
centered matrix, and project onto them. The harness allows `scikit-learn`, whose `decomposition.PCA` is
exactly this core — center, SVD of the centered matrix, deterministic `svd_flip`, projection onto the
top components, `random_state` for the randomized solver — so the fill is a two-line call to
`PCA(n_components=2, random_state=...)`; the derivation above is *why* that is the right first rung
rather than the random projection. The module is in the answer.

Now what this floor must do on the three datasets, because that is the whole reason to run it. PCA keeps
the directions of greatest *variance* and discards the rest. When the class neighborhoods live in the
top two variance directions, PCA keeps them and scores high; when they do not, PCA is blind by
construction, and there is exactly one reason it usually does not: PCA is *linear*. It can only place
points on a flat plane through the cloud. If the data lies on a curved manifold — digits whose
appearance varies along nonlinear modes, clothing whose pixel correlations bend — a flat projection
folds the curl and lands points from opposite sides of a fold adjacent. The local neighborhoods get
mangled; the global variance survives.

I can predict *which* metric moves and in which direction from their definitions, because the fold acts
asymmetrically. Trustworthiness punishes *false* neighbors the map invents, and a fold is a
false-neighbor factory, stacking two originally-distant sheets on top of each other. Continuity punishes
*tearing*, and a linear map is smooth — two genuine neighbors close in all coordinates including the two
PCA keeps land close. So the clean prediction: trustworthiness and especially kNN accuracy collapse
while continuity holds up relatively — a flat shadow of a curved cloud invents false neighbors and
smears the ten digit classes into one overlapping blob where a 7-NN classifier cannot separate them, but
it does not tear the coarse layout apart. On scale, a classifier that learned nothing sits at chance
≈ 1/10 on MNIST's ten classes; a linear shadow separates a few gross groups while smearing the rest, so
I expect a few times chance, in the low 0.3s. The three datasets should split: clothing categories carry
more coarse linear contrast (a boot's pixels differ from a shirt's along broad low-frequency directions
PCA keeps), so Fashion should do least badly on kNN, while 20-Newsgroups — already linearly pre-reduced
to 50 dimensions by truncated SVD, itself a linear operation close to PCA — gets a further linear
projection of an already-linear summary of diffuse, overlapping topics (chance ≈ 1/5), the weakest
setup relative to its room.

And because the whole failure is linearity and not sampling, PCA is deterministic up to the sign
convention, so the seed-to-seed spread should be tiny — any variance across the seeds can only come from
the scoring probe's stratified split, not the reducer, and large method variance here would mean I
misunderstood something. If I see the split I predict — high continuity, collapsed trustworthiness and
kNN, tight seeds — the diagnosis is settled: this is a *nonlinearity* problem, not a fitting problem.
The fix is not a better linear direction; the global layout PCA gives is the one thing a flat map gets
right and worth *keeping*. So the natural next move is to keep PCA's globally-faithful skeleton as a
starting point and refine it locally, bending neighborhoods to follow the manifold without discarding
the global frame.
