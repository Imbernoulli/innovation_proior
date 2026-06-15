Let me start from what actually goes wrong when I try to fit a cloud of correlated points. I have `n` individuals, each carrying `q` measured variables — stature and leg-length, or a handful of test scores, or several readings off one object — and I want a clean low-dimensional stand-in for where the data lives. The obvious tool is the regression line: pick `y` as the thing to predict, minimize `S(y' - y)^2` where `y'` is the line's ordinate above the observed `x`. Fine. But now I swap roles and regress `x` on `y`, minimizing `S(x' - x)^2`, and I get a *different* line through the very same points. Two lines, not one. And it's clear why: the regression residual is measured along one coordinate axis, so the answer secretly assumes that variable is exact and the other carries all the error. That's false here — a man at a given instant has one true position, but *both* his recorded time and his recorded position wobble from trial to trial. Every coordinate is on equal footing, so a criterion that privileges one of them is the wrong shape of answer. I need a fit that's symmetric in all `q` variables and returns one unique flat for the cloud.

What's a symmetric notion of "distance from a point to a line"? Not the vertical gap, not the horizontal gap — the *perpendicular* distance. That treats every direction alike; it doesn't care which axis I call dependent. So let me make the criterion the sum of squared perpendicular distances and minimize that. Call the perpendicular from point `P_k` to the line (or plane) `p_k`, and minimize
```
U = S(p^2).
```
This is symmetric by construction — rotate the coordinate frame however I like and `U` doesn't change, because perpendicular distance is a geometric quantity. Good. That's the right objective. Now I have to actually solve it.

Before grinding, one recognition that saves a lot. `U = S(p^2)` is exactly the second moment of the point system about the line — its moment of inertia if I load every point equally. And there's a fact from dynamics I can borrow wholesale: the second moment of a system about a family of *parallel* axes is least for the axis through the centroid. So whatever the best-fitting line's direction turns out to be, it must pass through the mean of the points. Let me confirm that falls out of the algebra rather than just asserting it. Write the plane as `l_1 x_1 + ... + l_q x_q = p` with the direction cosines normalized, `sum_i l_i^2 = 1`, and the perpendicular distance of a point is `(sum_i l_i x_i) - p`. So
```
U = S( (sum_i l_i x_i) - p )^2,
```
to be minimized over the `l_i` and `p` subject to `sum l_i^2 = 1`. Differentiate first with respect to `p` alone (it's unconstrained):
```
dU/dp = -2 S( sum_i l_i x_i - p ) = 0  =>  p = sum_i l_i * (S(x_i)/n) = sum_i l_i * xbar_i.
```
So `p = l . xbar`: the plane passes through the centroid, exactly as the inertia argument promised. From now on I can shift my origin to the centroid and drop the means; I'm only hunting for a *direction* `l`.

Now bring in the constraint with a Lagrange multiplier `Q` and vary the `l_u`. After substituting `p = l.xbar`, the centered objective is `U = S( sum_i l_i (x_i - xbar_i) )^2`, and `d/dl_u` of `U + Q(sum l^2 - 1)` set to zero gives, for each `u`,
```
sum_v l_v * S( (x_u - xbar_u)(x_v - xbar_v) ) + Q * l_u = 0.
```
The middle sum is just `n` times a covariance, `S((x_u-xbar_u)(x_v-xbar_v)) = n * sigma_u sigma_v r_uv`, so dividing by `n`,
```
sum_v ( sigma_u sigma_v r_uv ) l_v + (Q/n) l_u = 0     for every u.        (*)
```
That's a linear eigen-type system in `l` with the covariance matrix on the left and `-Q/n` playing the role of an eigenvalue. Let me pin down what `Q` is. Multiply equation `(*)` for index `u` by `l_u` and sum over `u`. The left double-sum `sum_{u,v} sigma_u sigma_v r_uv l_u l_v` is, term by term, exactly `U/n` (the mean squared residual is the same quadratic form in `l`), and the second piece is `(Q/n) sum_u l_u^2 = Q/n` by the constraint. So
```
U_m/n + Q/n = 0   =>   Q = -U_m,
```
with `U_m` the minimized value of `U`. So `Q/n = -U_m/n = -Sigma^2`, the mean square residual. The multiplier I introduced as a bookkeeping device turns out to *be* the residual variance, up to a sign. That's the first sign that the constrained optimum is secretly an eigenproblem whose eigenvalue carries the meaning of the fit.

Let me make that explicit. Equation `(*)` says: the covariance matrix times the direction `l` equals `Sigma^2` times `l` (after moving the sign around — `Q/n = -Sigma^2`). Writing the covariance matrix as `C` with entries `C_uv = sigma_u sigma_v r_uv`, the stationarity condition is `C l = Sigma^2 l`. So `l` is an eigenvector of the covariance matrix and `Sigma^2` is the corresponding eigenvalue. For a *plane*, the normal direction is the direction I am throwing away, and the residual is the variance along that normal. I want that residual `Sigma^2` minimal, so I take the *least* root of the determinantal equation `det(C - Sigma^2 I) = 0`; the residual is the smallest eigenvalue and `l` its eigenvector. The determinant version, dividing rows and columns through by the sigmas to get it in correlation form, is
```
det( [ 1 - Sigma^2/sigma_1^2 ,  r_12 , ... ; r_21 , 1 - Sigma^2/sigma_2^2 , ... ; ... ] ) = 0,
```
and I choose the least root because the residual must be as small as possible. That's the complete analytical solution for the best-fitting plane: it depends on the data only through the means (to locate the centroid), the standard deviations, and the correlations — nothing else.

But I actually care more about the best-fitting *line* than the best-fitting plane, because the line is the one-dimensional summary, the dominant direction of the cloud. So let me redo the perpendicular-distance minimization for a line in `q`-space, direction cosines `l`, through the centroid. The squared perpendicular distance of a centered point `(x - xbar)` from the line through the origin with unit direction `l` is `|x - xbar|^2 - (l . (x - xbar))^2` — the full length minus the part along the line, by Pythagoras. Sum and divide by `n`:
```
Sigma'^2 = (1/n) S( |x - xbar|^2 ) - (1/n) S( (l.(x - xbar))^2 )
         = ( sigma_1^2 + sigma_2^2 + ... + sigma_q^2 ) - [ variance of the data projected onto l ].
```
Stare at this. The first bracket, the total of the per-coordinate variances, is a *constant* — it doesn't depend on `l` at all. The second bracket is the variance of the data *along* the line `l`. So minimizing the perpendicular residual of the line is identical to *maximizing the variance of the projection onto the line*. The two problems are the same problem. Minimize scatter off the line ⟺ maximize scatter along it, because total scatter splits, Pythagorean, into "along" plus "perpendicular," and the total is fixed. I didn't expect the least-squares fit and a maximize-variance criterion to coincide, but the constant-total identity forces it. And the line that maximizes the projected variance is the eigenvector of the covariance matrix belonging to the *greatest* eigenvalue — the mirror image of the plane case, where I took the least. The best-fitting line is the direction of greatest variance; the best-fitting plane is perpendicular to the direction of least variance.

Geometrically this is the principal-axis story of the correlation ellipsoid, with one easy place to get turned around. The contours of a normal correlated cloud are concentric similar ellipsoids,
```
x^2/sigma_x^2 + y^2/sigma_y^2 - 2 r_xy xy/(sigma_x sigma_y) = const     (in 2D),
```
whose quadratic form uses the inverse covariance, so the contour axes are the same eigenvectors as `C` but with reciprocal axis weights. The variance along a direction is the direct covariance quadratic form `l^T C l`. For a plane, I choose the normal direction with the smallest value of `l^T C l`, because that value is the residual variance I leave perpendicular to the plane. For a line, I choose the line direction with the largest value of `l^T C l`, because that value is the variance I keep along the line. So the two-dimensional picture is no longer ambiguous: the fitted line is the major axis of the correlation ellipse, and the discarded normal is the minor axis. I can read off the angle directly: maximizing the projected variance of the correlation ellipse gives
```
tan 2*theta = 2 r_xy sigma_x sigma_y / (sigma_x^2 - sigma_y^2),
```
which is exactly the major axis of the correlation ellipse. So in 2D my "line of the cloud" *is* the major axis of the bivariate normal contour — a single, symmetric, well-defined direction, finally, instead of two regression lines. And there's a clean degenerate case that tells me when the problem has no answer: if the ellipsoid is a sphere — all correlations zero and all sigmas equal — every direction is equally good, there's no unique best fit. That's isotropy; the method correctly refuses to pick a direction when the data has no preferred one.

Now let me come at the *reductive* problem from the other side, because I want more than one direction — I want an ordered sequence of new variables. Forget geometry for a second and think in terms of variance accounting. Suppose I look for new variables `gamma_1, gamma_2, ...` that are mutually uncorrelated, each of unit variance, such that I can write each observed (standardized) variable as a linear combination `z_i = sum_j a_ij gamma_j`. I'll choose the first new variable `gamma_1` to *account for as much of the total variance of the `z`'s as possible*, then `gamma_2` to account for as much of what's left, and so on. How much of `z_i`'s variance does `gamma_1` explain? Since the `gamma`'s are uncorrelated and unit-variance, squaring `z_i = sum_j a_ij gamma_j` and taking the mean gives `var(z_i) = sum_j a_ij^2`, and the part from `gamma_1` is `a_i1^2`. Summed over all the `z`'s, the total variance `gamma_1` explains is
```
S = a_11^2 + a_21^2 + ... + a_q1^2 = sum_i a_i1^2.
```
I want to maximize `S` over the loadings `a_i1`, subject to the constraint that the model reproduces the observed correlations — `sum_s a_is a_js = r_ij`, or `R = A A^T`. Let me write the same condition in a way I can differentiate without hiding the moving parts. If the first component is a unit-variance linear score `gamma = w^T z`, then
```
var(gamma) = w^T R w = 1,
```
and its loading on the observed variables is the covariance vector
```
a = E[z gamma] = E[z z^T] w = R w.
```
The total variance this component accounts for is therefore
```
S = a^T a = w^T R^2 w.
```
Now maximize `w^T R^2 w` subject to `w^T R w = 1`. The Lagrangian derivative is
```
R^2 w - k R w = 0.
```
Since `a = R w`, this says the loading vector itself obeys
```
sum_j r_ij a_j = k a_i     for every i,        i.e.   R a = k a,
```
where `R` is the correlation matrix. So again — an eigenvalue problem, now on the correlation matrix `R`, with eigenvector `a` (the loadings of the first component) and eigenvalue `k`. And `S`, the variance I was maximizing, equals `k`: from the stationarity equation, `S = w^T R^2 w = k w^T R w = k`. So to make `gamma_1` explain the most variance I take the *greatest* root `k_1` of
```
det(R - k I) = 0,
```
and its eigenvector gives the loadings. This is the same determinantal "characteristic equation" Cauchy studied for the principal axes of quadrics — and Cauchy already proved that for a symmetric matrix every root of this equation is real, so I'm never going to be handed a complex variance. Better: the roots of `R`'s characteristic equation are not just real but non-negative, because each is a variance (a sum of squares) and a variance can't be negative — a negative root would make the constant-density locus a hyperboloid running off to infinity with infinite total probability, which is absurd for a genuine distribution. Real, non-negative, and ordered: exactly what an "importance ordering" of components needs.

So the two routes — minimize perpendicular residual (the geometric, least-squares route) and maximize explained variance (the accounting route) — land on the *same* eigenproblem of the second-moment matrix. That's the confirmation I wanted that I've found the right object and not just one of several. The first new variable is the top eigenvector of the covariance (or correlation) matrix; its eigenvalue is the variance it captures.

One detail I glossed: covariance matrix or correlation matrix? It depends on the units. If the `q` variables are in comparable, meaningful units, the natural metric is Euclidean in the raw coordinates and I diagonalize the covariance. If they're in arbitrary, incommensurable units — test "scores" with no shared scale — then a rescaling of one variable would arbitrarily squeeze or stretch the ellipsoid and move the principal axes, so I first standardize each variable to zero mean and unit variance, `z_i = (x_i - xbar_i)/sigma_i`, and diagonalize the correlation matrix instead. The method needs a *metric* — a notion of distance in the `q`-space — and the choice of covariance-vs-correlation is exactly the choice of that metric. There's no universal right answer; it's a modeling decision about what "equal importance" means. I'll keep the default as centering (the covariance route), which is what you want when the coordinates already share a scale, and note that standardizing first is the correlation route.

Now I need the *rest* of the components, not just the first. The recipe writes itself from the accounting view: once `gamma_1` is fixed, the next component should account for as much of the *residual* variance as possible — the variance left after `gamma_1`'s contribution is removed. The contribution of `gamma_1` to the cross-structure is the rank-one piece `a_i1 a_j1` (the outer product of the first loading vector), and the residual structure is what's left after I strip it out:
```
R' = R - a_1 a_1^T,
```
where the loading vector `a_1` is scaled so `a_1^T a_1 = k_1`. This is precisely the correlation/covariance matrix of the residuals `z_i - a_i1 gamma_1`, the part of the data uncorrelated with `gamma_1`. Then `gamma_2` is the top eigenvector of `R'`, `gamma_3` the top of `R''`, and so on — peel off one rank-one layer at a time. This *deflation* is just the statement that the eigenvectors of a symmetric matrix are mutually orthogonal: maximizing variance over the subspace orthogonal to `gamma_1` returns the next eigenvector, with eigenvalue the next-largest root. So I don't even have to deflate in principle — the full eigen-decomposition of `R` hands me all the components at once, ordered by eigenvalue. But deflation is how I'd compute them one at a time if I only want the top few. Roundoff in a trial vector is damped by the next multiplication, and the running sum of the extracted roots checks the calculation against the known total. Orthogonality among the loadings is automatic: `a_h^T a_m = delta_{hm} k_h`, columns orthogonal, lengths the square roots of the eigenvalues.

How much of the whole picture have I kept after `m` components? The total variance is the sum of all the eigenvalues — for the correlation matrix that's `q` (the trace, ones on the diagonal) — so the fraction of total variance carried by the `i`-th component is `k_i / (sum of all roots)`, and the cumulative fraction `(k_1 + ... + k_m)/(sum)` tells me how faithful an `m`-dimensional summary is. That's the natural stopping rule: keep enough components to capture the variance I care about, and the ones I drop are the small-eigenvalue directions where the cloud is thin — noise and redundancy.

Now, computing this. Solving `det(R - k I) = 0` explicitly — expanding the determinant into its characteristic polynomial `(-1)^q (k^q - q k^{q-1} + S_2 k^{q-2} - ...)` and rooting it, then back-substituting into the homogeneous system for each eigenvector — is brutal by hand for any real `q`. I want something cheaper that gives me the *largest* root and its eigenvector first, since those are the ones I keep, and lets me stop early. I can avoid the determinant by taking any trial direction `a` and hitting it with the matrix, `a' = R a`. Geometrically `R` stretches space most along its dominant principal axis, so `a'` is `a` rotated *toward* the top eigenvector. Iterate: `a -> R a -> R^2 a -> ...`. Expand the trial vector in the eigenbasis, `a = sum_j c_j v_j`; then `R^t a = sum_j c_j k_j^t v_j`, and since `k_1` is the largest root, `k_1^t` dominates every other `k_j^t` geometrically, so the iterate aligns with `v_1` and the per-step scale factor converges to `k_1`. Concretely: substitute trial values `a_1, ..., a_q`, multiply by the rows of `R` to get `a'`, divide all the `a'` by one of them (or by their previous values) to read off the common ratio — that ratio is converging to `k_1` and the normalized vector to the top eigenvector. Repeat until the digits stop moving, normalize so `sum a_i^2 = k_1`, and I have the first component without ever expanding a determinant. The only way it fails is if the starting vector has zero component along `v_1`, an exceptional case I can avoid by changing the trial values. Then deflate — subtract `a_1 a_1^T` from the matrix — and run the same iteration on the reduced matrix for the second component, and so on. The running sum of the roots I've extracted, compared against the known total, tells me at every stage how much variance I've accounted for, so I know exactly when to stop. This power-iteration-plus-deflation is the hand-computable engine.

Now the computational object is clear: center the data, form the second-moment structure, take its top eigenvectors (greatest-variance directions, equivalently the subspace with least perpendicular residual), and project the data onto the first few. I still have to be careful about *how* to compute those directions numerically, because forming the covariance matrix and diagonalizing it has a cost. Working with `n` samples and `q` features, the centered data is a matrix `Xc` of shape `(n, q)`. The covariance is `C = (1/(n-1)) Xc^T Xc`. I *could* form `C` (a `q x q` matrix) and call a symmetric eigensolver `eigh(C)`. But forming `Xc^T Xc` squares the condition number of `Xc`: small singular values get squared and drown in roundoff, and I lose accuracy precisely on the small-variance directions. The fix is the singular value decomposition of the centered data directly. Factor
```
Xc = U S V^T,
```
with `U` `(n x r)` orthonormal, `S` the diagonal of singular values (descending), `V` `(q x r)` orthonormal. Then
```
Xc^T Xc = V S U^T U S V^T = V S^2 V^T,
```
so the right singular vectors `V` are exactly the eigenvectors of `Xc^T Xc`, hence of the covariance `C` — my principal directions — and the eigenvalue (variance) of the `i`-th component is
```
explained_variance_i = S_i^2 / (n - 1).
```
The SVD gives me the same `V` and the same ordering (singular values come sorted, so I don't even have to sort eigenvalues), but it never forms the ill-conditioned Gram matrix, so it's the numerically sound way to do the exact same linear algebra. The power iteration I derived by hand is the small-by-hand version of this; the SVD is the industrial version, and they compute the identical object.

The projection onto the top `k` components is then `Xc @ V[:, :k]`, the coordinates of each centered point in the principal-axis frame. And there's a tiny but real reproducibility wrinkle: the sign of each singular vector is arbitrary — `U S V^T = (-U) S (-V)^T` is the same factorization — so two runs (or two LAPACK builds) can flip the sign of a component, flipping that coordinate's axis. To make the output deterministic I fix each right singular vector's sign by a fixed rule, the same kind of row-of-`Vt` decision used in the full-SVD solver: force the largest-magnitude entry of that row to be positive. With a low-dimensional target like a 2D embedding, I just keep the first two columns. Note that the projected coordinates `Xc @ V[:, :k]` are exactly the first `k` columns of `U S` — `Xc V = U S V^T V = U S` — so I can read the embedding straight off the SVD without a separate matmul.

In code I solve the centered cloud by a full SVD, keep the right singular vectors as directions, and return the top columns of `U S` as the same coordinates as `Xc @ Vt[:k].T`:

```python
import numpy as np
from numpy.typing import NDArray
from scipy import linalg


class PCA:
    """Principal Component Analysis. Project centered data onto the top-k
    directions of greatest variance == the least-perpendicular-residual flat ==
    the top eigenvectors of the covariance, computed via the SVD of the
    centered data (numerically sound; no explicit Gram matrix)."""

    def __init__(self, n_components: int = 2, random_state: int | None = None):
        self.n_components = n_components
        self.random_state = random_state   # only used by approximate/randomized solvers

    @staticmethod
    def _svd_flip(U: NDArray[np.float64], Vt: NDArray[np.float64]):
        # Singular-vector signs are arbitrary; fix them deterministically so the
        # embedding is reproducible. Match the full-SVD convention that decides
        # from the largest-magnitude entry in each row of Vt.
        max_abs = np.argmax(np.abs(Vt), axis=1)
        signs = np.sign(Vt[np.arange(Vt.shape[0]), max_abs])
        return U * signs[np.newaxis, :], Vt * signs[:, np.newaxis]

    def _solve_centered(self, Xc: NDArray[np.float64]):
        # Full SVD of centered data, not eig(Xc.T @ Xc): Xc = U S Vt and
        # Xc.T @ Xc = V S^2 V.T, so rows of Vt are principal directions.
        U, S, Vt = linalg.svd(Xc, full_matrices=False)
        U, Vt = self._svd_flip(U, Vt)
        return U, S, Vt

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        k = self.n_components

        # Remove the location nuisance: the best-fitting flat passes through the
        # centroid (second moment is least about the mean), so center the data.
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        U, S, Vt = self._solve_centered(Xc)

        # variance captured by each component: eigenvalue of the covariance = S^2/(n-1);
        # fraction of total variance = explained_variance / sum(explained_variance).
        self.components_ = Vt[:k]                                  # top-k principal axes
        self.explained_variance_ = (S[:k] ** 2) / (n_samples - 1)
        self.singular_values_ = S[:k]
        total_var = (S**2).sum() / (n_samples - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / total_var

        # Re-express the centered cloud in the principal-axis frame:
        # Xc @ components_.T == (U S)[:, :k], the top-k principal coordinates.
        return U[:, :k] * S[:k]
```

I have the chain now. The regression line is asymmetric — it gives two different lines depending on which variable I call dependent, since it minimizes residuals along one axis and presumes that axis error-free, which is wrong when every coordinate is observed with error. Switching to the symmetric criterion, minimizing the sum of squared *perpendicular* distances, fixes that; the inertia argument forces the fit through the centroid, so after centering I am only choosing a direction. The Lagrange-constrained minimization turns into the eigenproblem of the second-moment matrix, with `Q/n = -Sigma^2`, so the multiplier itself is the residual variance up to sign. For a line, the Pythagorean split of total scatter into "along" plus "perpendicular," with the total fixed, makes minimizing perpendicular residual identical to maximizing projected variance — so the best-fitting line is the top eigenvector of the covariance, and in 2D it is the major axis of the correlation ellipse. Coming the other way, choosing uncorrelated unit-variance components to explain the most total variance in order leads to the same eigenproblem on the correlation matrix, with the characteristic equation `det(R - kI)=0` whose roots Cauchy showed are real and which here are non-negative variances. Successive components peel off rank-one layers, `R - a_1 a_1^T` in the loading normalization `a_1^T a_1 = k_1`, and the fraction of total variance `k_i/sum k` gives the stopping rule. The power iteration extracts the top eigenvector and root cheaply and lets me stop early; its industrial form is the SVD of the centered data, which computes the identical principal directions `V` (right singular vectors) and variances `S^2/(n-1)` without ever forming the ill-conditioned Gram matrix. Fix the singular-vector signs for reproducibility, keep the top `k` columns, and the embedding `Xc V[:, :k] = (US)[:, :k]` is the data re-expressed on its own dominant directions.
