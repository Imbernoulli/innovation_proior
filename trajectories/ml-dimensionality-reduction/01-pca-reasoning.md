The embedding is the whole point, but it bolts onto a fixed scoring harness, and the floor I have to
start from is the cheapest faithful projection I can write into the scaffold's `fit_transform`. The
scaffold default hands me a random linear projection: orthonormalize a couple of Gaussian axes and
project. That is the Johnson–Lindenstrauss construction, and at a target of two dimensions it is a
genuinely poor floor — JL only promises distance preservation when the target dimension is large
enough to host the random directions, and two is nowhere near that, so a random pair of axes captures
whatever arbitrary slice of the variance it happens to land on, usually a thin one. So the first real
question is the right *first rung*: what is the cheapest projection that is not arbitrary — that picks
its directions from the data instead of from a random number generator — and what should I expect it
to do and to fail at on these three datasets, because that failure is what forces the second rung.

Let me write down what actually goes wrong with the naive linear fits the first rung has to beat, so I
know what shape of answer I want. Take the cloud of standardized points the harness hands me and ask
for a line through it. The obvious tool is the regression line: pick one coordinate as the thing to
predict, minimize the sum of squared residuals measured *along that coordinate's axis*. Fine — but now
swap which coordinate I call dependent and minimize residuals along the *other* axis, on the very same
points, and I get a different line. Two lines, not one, for one cloud. The reason is structural: the
regression residual is measured along a single coordinate direction, so the criterion secretly assumes
that coordinate is exact and the other carries all the error. On standardized pixel data that is
false — every coordinate is a measurement with its own wobble, none is privileged. A criterion that
privileges one axis is the wrong *shape* of answer. I need a fit that is symmetric in all the
coordinates and returns one unique flat for the cloud.

What is a symmetric notion of "distance from a point to a line"? Not the vertical gap, not the
horizontal gap — the *perpendicular* distance. That treats every direction alike; it does not care
which axis I call dependent. So make the criterion the sum of squared perpendicular distances and
minimize it. Call the perpendicular from point P_k to the fitted line p_k, and minimize U = sum of
p_k^2. This is symmetric by construction: rotate the coordinate frame however I like and U does not
change, because perpendicular distance is a geometric quantity. That is the right objective; now I have
to solve it, and the solution is going to tell me everything about why this rung behaves the way it
does.

One recognition saves a lot of grinding. U is exactly the second moment of the point system about the
line — its moment of inertia with every point weighted equally. And there is a fact I can borrow
wholesale from dynamics: the second moment of a system about a family of *parallel* axes is least for
the axis through the centroid. So whatever the best-fitting line's direction turns out to be, it must
pass through the mean of the points. Let me confirm it falls out of the algebra rather than just
asserting it. Write the flat as l·x = p with the direction normalized (||l|| = 1); the perpendicular
distance of a point is l·x − p. Differentiate U with respect to p alone (it is unconstrained): the
derivative is −2·sum(l·x − p) = 0, giving p = l·xbar. So the flat passes through the centroid, exactly
as the inertia argument promised. From here I shift the origin to the centroid and drop the means; I am
only hunting for a *direction*. This is the first concrete instruction the rung gives me: **center the
data**, then find directions.

Now bring in the unit-length constraint with a Lagrange multiplier Q and vary the direction. After
substituting p = l·xbar, the centered objective is U = sum of (l·(x − xbar))^2, and setting the
derivative of U + Q(||l||^2 − 1) to zero gives, component by component, the covariance matrix times l
equal to a scalar times l. Writing the covariance matrix C with entries scaled by the standard
deviations and correlations, stationarity is C l = sigma^2 l: l is an eigenvector of the covariance
matrix and the eigenvalue is the residual variance left perpendicular to the flat. Let me pin down what
Q is, because it carries the meaning. Multiply the stationarity equation by l and sum: the left side is
the mean squared residual U_m/n, the right is (Q/n)·||l||^2 = Q/n, so Q = −U_m. The multiplier I
introduced as bookkeeping turns out to *be* the residual variance, up to sign. For a plane I want that
residual minimal, so I take the *least* eigenvalue and its eigenvector as the discarded normal
direction.

But I care more about the best-fitting *line* — the one-dimensional summary, the dominant direction of
the cloud. Redo the perpendicular minimization for a line through the centroid: the squared
perpendicular distance of a centered point from the line with unit direction l is, by Pythagoras, the
full squared length minus the squared projection onto l. Sum and divide by n: the perpendicular
residual equals (total per-coordinate variance) minus (variance of the data projected onto l). Stare
at this. The first term, the total of the per-coordinate variances, is a *constant* — it does not
depend on l at all. The second term is the variance of the data *along* the line. So minimizing the
perpendicular residual of the line is identical to *maximizing the variance of the projection onto the
line*. The two problems are the same problem: minimize scatter off the line is the same as maximize
scatter along it, because total scatter splits Pythagoreanly into "along" plus "perpendicular" and the
total is fixed. And the line that maximizes projected variance is the eigenvector of the covariance
matrix with the *greatest* eigenvalue — the mirror image of the plane case. The best-fitting line is
the direction of greatest variance; the best-fitting plane is perpendicular to the direction of least
variance. In two coordinates this is precisely the major axis of the correlation ellipse: a single,
symmetric, well-defined direction, instead of two regression lines. There is even a clean degenerate
case that tells me when the answer is non-unique: if the cloud is an isotropic ball — all correlations
zero, all variances equal — every direction is equally good and there is no preferred axis. The method
correctly refuses to pick a direction when the data has none.

Come at the reductive problem from the accounting side now, because I want an *ordered sequence* of new
coordinates, not just one. Look for new variables that are mutually uncorrelated and unit-variance,
such that each observed standardized coordinate is a linear combination of them; choose the first new
variable to account for as much of the total variance as possible, the second for as much of what is
left, and so on. If the first component is a unit-variance score w·z, its loading on the observed
variables is the covariance vector a = C w, and the total variance it accounts for is a·a = w^T C^2 w.
Maximize that subject to w^T C w = 1, and the Lagrange condition collapses to C a = k a — again an
eigenproblem, with the loading vector the eigenvector and the eigenvalue k equal to the variance
accounted for. So the greatest eigenvalue's eigenvector is the first component, and the determinantal
characteristic equation det(C − kI) = 0 is the same one Cauchy studied for the principal axes of
quadrics; its roots are real (symmetric matrix) and here non-negative, because each root is a variance
and a variance cannot be negative. Real, non-negative, ordered — exactly what an importance ranking of
components needs. The two routes, minimize perpendicular residual and maximize explained variance, land
on the *same* eigenproblem of the second-moment matrix. That is the confirmation I have the right
object and not one of several.

Successive components write themselves: once the first is fixed, the next should explain as much of the
*residual* variance as possible, which is the top eigenvector of the deflated matrix C − k_1 v_1 v_1^T —
and eigenvector orthogonality makes this automatic, so the full eigendecomposition hands me every
component at once, ordered by eigenvalue. The fraction of total variance carried by the first m
components, (k_1 + ... + k_m) divided by the sum of all eigenvalues, is the natural stopping rule. For
a 2D embedding I simply keep the top two.

Now the computation, because *how* I compute the directions matters numerically and the scaffold's
budget is real. I could form the covariance C = Xc^T Xc and call a symmetric eigensolver. But forming
Xc^T Xc squares the condition number of Xc: small singular values get squared and drown in roundoff, so
I lose accuracy precisely on the small-variance directions. The numerically sound move is the singular
value decomposition of the centered data directly, Xc = U S V^T. Then Xc^T Xc = V S^2 V^T, so the right
singular vectors V are exactly the eigenvectors of the covariance — my principal directions — the
variance of component i is S_i^2/(n−1), and because singular values come sorted I do not even have to
sort eigenvalues. Better, the embedding is the top columns of U S, since Xc V = U S: I read the
coordinates straight off the factorization. There is one reproducibility wrinkle the harness cares
about: singular-vector signs are arbitrary (U S V^T = (−U) S (−V)^T), so two runs could flip a
coordinate's axis; fixing each vector's sign by a deterministic rule (force the largest-magnitude entry
positive) makes the output stable across the seeds.

That is the entire first rung: center the standardized cloud, take the top two right singular vectors of
the centered matrix, and return the projection onto them. In the scaffold this is the smallest possible
edit — `fit_transform` builds the SVD-backed principal-component projection and returns the first two
principal coordinates of `X`. I do not need to hand-roll the SVD and sign-flip: the harness allows
`scikit-learn`, whose `decomposition.PCA` solver is exactly this linear-algebra core — center, full
(or randomized) SVD of the centered matrix, deterministic `svd_flip` from the rows of the loadings,
projection onto the top components, and it threads `random_state` for the randomized solver so the
embedding is reproducible across the three seeds. So the rung-1 fill is a two-line call to
`PCA(n_components=2, random_state=...)`; the derivation above is *why* that call is the right first
rung rather than the random projection. The distilled module is in the answer.

Now reason about what this floor must do on the three datasets, because that is the whole reason to run
it. PCA keeps the directions of greatest *variance* and throws away the rest. When the structure I care
about — the class neighborhoods — lives in the top two variance directions, PCA will keep it and the
scores will be high. When it does not, PCA is blind to it by construction, and there is exactly one
reason it usually does not: PCA is *linear*. It can only place points on a flat two-dimensional plane
through the cloud. If the data lies on a curved manifold — digits whose appearance varies smoothly along
nonlinear modes, clothing images whose pixel correlations bend — then a flat projection folds the curl
and lands points from opposite sides of a fold right next to each other. The neighborhoods get mangled;
the global variance survives. So I expect the continuity scores to hold up reasonably (PCA does not
*tear* nearby points apart — it preserves the coarse layout) while the trustworthiness and especially
the kNN accuracy collapse, because a flat shadow of a curved cloud invents false neighbors and smears
the ten digit classes into one overlapping blob where a 7-NN classifier cannot separate them.

The three datasets should not behave identically, and the split is informative. MNIST and Fashion-MNIST
are 784-dimensional images whose true structure is strongly nonlinear, so I expect PCA's class
preservation there to be weak — the digit and garment classes are not separable in any two linear
directions. 20-Newsgroups is already linearly pre-reduced to 50 dimensions by truncated SVD before the
harness ever sees it, which is itself a linear operation closely related to PCA, so the further drop to
two dimensions is again a linear projection of an already-linear summary; its class structure (five
topic groups) is more diffuse and overlapping, so I expect that to be the weakest of all on kNN
accuracy. Across the board, this is the floor: the cheapest non-arbitrary projection, and the cheapest
to *diagnose*, because whatever it loses it loses for one clean reason — linearity.

The falsifiable expectation I will check against the measured numbers: PCA should beat the random
projection comfortably (it picks data-driven directions), but its kNN accuracy should land far below any
nonlinear neighbor-embedding method — somewhere in the 0.3 range on the image datasets and lower on
newsgroups — with trustworthiness dragged down with it, while continuity stays the highest-relative of
its own three metrics because the global layout is the one thing a linear map preserves. If that is what
I see, then the diagnosis is settled and points straight at the next rung: this is a *nonlinearity*
problem, not a fitting problem. The fix is not a better linear direction; the global layout PCA gives
is actually worth *keeping* — it is the one thing a flat map gets right — so the natural next move is to
keep PCA's globally-faithful skeleton as a starting point and *refine it locally*, bending the
neighborhoods to follow the manifold without discarding the global frame. That refine-from-PCA strategy,
driven by relative-order triplet forces, is exactly what step 2 builds.
