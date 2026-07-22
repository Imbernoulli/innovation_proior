Let me start from what actually goes wrong when I try to fit a cloud of correlated points. I have `n` individuals, each carrying `q` measured variables — stature and leg-length, or a handful of test scores, or several readings off one object — and I want a clean low-dimensional stand-in for where the data lives. The obvious tool is the regression line: pick `y` as the thing to predict, minimize `S(y' - y)^2` where `y'` is the line's ordinate above the observed `x`. Fine. But now I swap roles and regress `x` on `y`, minimizing `S(x' - x)^2`, and I get a *different* line through the very same points. Two lines, not one. And it's clear why: the regression residual is measured along one coordinate axis, so the answer secretly assumes that variable is exact and the other carries all the error. That's false here — a man at a given instant has one true position, but *both* his recorded time and his recorded position wobble from trial to trial. Every coordinate is on equal footing, so a criterion that privileges one of them is the wrong shape of answer. I need a fit that's symmetric in all `q` variables and returns one unique flat for the cloud.

What's a symmetric notion of "distance from a point to a line"? Not the vertical gap, not the horizontal gap — the *perpendicular* distance. That treats every direction alike; it doesn't care which axis I call dependent. So let me make the criterion the sum of squared perpendicular distances and minimize that. Call the perpendicular from point `P_k` to the line (or plane) `p_k`, and minimize
```
U = S(p^2).
```
This is symmetric by construction — rotate the coordinate frame however I like and `U` doesn't change, because perpendicular distance is a geometric quantity. Now solve it.

One recognition saves a lot of grinding. `U = S(p^2)` is exactly the second moment of the point system about the line — its moment of inertia if I load every point equally. And there's a fact from dynamics I can borrow wholesale: the second moment of a system about a family of *parallel* axes is least for the axis through the centroid. So whatever the best-fitting line's direction turns out to be, it must pass through the mean of the points. The algebra confirms it directly. Write the plane as `l_1 x_1 + ... + l_q x_q = p` with the direction cosines normalized, `sum_i l_i^2 = 1`, and the perpendicular distance of a point is `(sum_i l_i x_i) - p`. So
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
That's a linear system in `l` with the covariance matrix on the left and `-Q/n` sitting where an eigenvalue would sit. What is `Q`? Right now it's an undetermined bookkeeping symbol; multiply equation `(*)` for index `u` by `l_u` and sum over `u`. The left double-sum `sum_{u,v} sigma_u sigma_v r_uv l_u l_v` is, term by term, exactly `U/n` (the mean squared residual is the same quadratic form in `l`), and the second piece is `(Q/n) sum_u l_u^2 = Q/n` by the constraint. So
```
U_m/n + Q/n = 0   =>   Q = -U_m,
```
with `U_m` the minimized value of `U`. So `Q/n = -U_m/n = -Sigma^2`, the mean square residual. The multiplier I introduced as a placeholder turns out to *be* the residual variance, up to a sign — it carries the meaning of the fit.

Writing the covariance matrix as `C` with entries `C_uv = sigma_u sigma_v r_uv`, equation `(*)` then reads `C l = Sigma^2 l` (after moving the sign around with `Q/n = -Sigma^2`). So the stationary directions are eigenvectors of the covariance matrix, and the eigenvalue `Sigma^2` is the residual variance left along that direction. The identity `Q/n = -Sigma^2` is the linchpin of the whole reduction, so I check it on numbers before building on it. A `50`-point cloud in `3`-space with a deliberately anisotropic spread, centered: the smallest eigenvalue of `C` comes out `0.28782`; the variance of the points' offsets along that same eigen-direction, `var(Xc · normal)`, comes out `0.28782` to every digit shown. The eigenvalue genuinely is the residual variance, not just a quantity I labelled that way. For a *plane*, the normal direction is the direction I am throwing away, and I want that residual minimal, so I take the *least* root of `det(C - Sigma^2 I) = 0`; the residual is the smallest eigenvalue and `l` its eigenvector. The determinant version, dividing rows and columns through by the sigmas to get it in correlation form, is
```
det( [ 1 - Sigma^2/sigma_1^2 ,  r_12 , ... ; r_21 , 1 - Sigma^2/sigma_2^2 , ... ; ... ] ) = 0,
```
and I choose the least root because the residual must be as small as possible. That's the complete analytical solution for the best-fitting plane: it depends on the data only through the means (to locate the centroid), the standard deviations, and the correlations — nothing else.

But I actually care more about the best-fitting *line* than the best-fitting plane, because the line is the one-dimensional summary, the dominant direction of the cloud. So let me redo the perpendicular-distance minimization for a line in `q`-space, direction cosines `l`, through the centroid. The squared perpendicular distance of a centered point `(x - xbar)` from the line through the origin with unit direction `l` is `|x - xbar|^2 - (l . (x - xbar))^2` — the full length minus the part along the line, by Pythagoras. Sum and divide by `n`:
```
Sigma'^2 = (1/n) S( |x - xbar|^2 ) - (1/n) S( (l.(x - xbar))^2 )
         = ( sigma_1^2 + sigma_2^2 + ... + sigma_q^2 ) - [ variance of the data projected onto l ].
```
The first bracket, the total of the per-coordinate variances, is a *constant* — it doesn't depend on `l` at all. The second bracket is the variance of the data *along* the line `l`. So minimizing the perpendicular residual of the line should be identical to *maximizing the variance of the projection onto the line*: minimize scatter off the line ⟺ maximize scatter along it, because total scatter splits, Pythagorean, into "along" plus "perpendicular," and the total is fixed. That's a strong claim resting on the split being an exact identity for *every* direction, not just the optimum, so I test it on the same `3`-space cloud. The total of per-coordinate variances is `6.3620` (and equals `trace(C)`, as it must). Picking four directions — the top eigenvector, a middling eigenvector, a raw axis `(1,0,0)`, and a random tilt — and computing "along" `= var(Xc·l)` and "perpendicular" `= mean squared distance off the line` separately for each: I get `(5.284, 1.078)`, `(0.790, 5.572)`, `(4.997, 1.365)`, `(1.099, 5.263)`. Every pair sums to `6.362`, the same constant, regardless of the direction. So the split is an exact identity and the trade-off is rigid: pushing "along" up pushes "perpendicular" down by exactly as much. Going in I didn't expect the least-squares fit and a maximize-variance criterion to coincide, but the fixed total forces it. And the line that maximizes the projected variance is the eigenvector of the covariance matrix belonging to the *greatest* eigenvalue — the mirror image of the plane case, where I took the least. (In the run above, "along" for the top eigenvector, `5.284`, equals its eigenvalue to the digits printed, which is the same statement.) The best-fitting line is the direction of greatest variance; the best-fitting plane is perpendicular to the direction of least variance.

Geometrically this is the principal-axis story of the correlation ellipsoid, with one easy place to get turned around. The contours of a normal correlated cloud are concentric similar ellipsoids,
```
x^2/sigma_x^2 + y^2/sigma_y^2 - 2 r_xy xy/(sigma_x sigma_y) = const     (in 2D),
```
whose quadratic form uses the inverse covariance, so the contour axes are the same eigenvectors as `C` but with reciprocal axis weights. The variance along a direction is the direct covariance quadratic form `l^T C l`. For a plane, I choose the normal direction with the smallest value of `l^T C l`, because that value is the residual variance I leave perpendicular to the plane. For a line, I choose the line direction with the largest value of `l^T C l`, because that value is the variance I keep along the line. So the two-dimensional picture is no longer ambiguous: the fitted line ought to be the major axis of the correlation ellipse, and the discarded normal its minor axis.

"The major axis" is easy to assert and easy to get rotated wrong by 90 degrees, so I make it concrete. Take `sigma_x = 2`, `sigma_y = 1`, `r_xy = 0.6`, so
```
C = [[4, 1.2], [1.2, 1]].
```
A direction at angle `theta` is `l = (cos theta, sin theta)`, and its projected variance is `l^T C l = 4 cos^2 + 2.4 cos sin + sin^2`. Differentiating in `theta` and setting to zero — `d/dtheta` of `l^T C l` is `2 l^T C l_perp` — gives the stationarity condition `(sigma_x^2 - sigma_y^2) sin 2theta = 2 r_xy sigma_x sigma_y cos 2theta`, i.e.
```
tan 2*theta = 2 r_xy sigma_x sigma_y / (sigma_x^2 - sigma_y^2).
```
With my numbers, `tan 2theta = 2(0.6)(2)(1)/(4 - 1) = 2.4/3 = 0.8`, so `2theta = 38.66°` and `theta = 19.33°`. Now I want to know two things: is this the *max* (the major axis) and not the perpendicular min, and does it actually agree with the eigenvector. Diagonalizing `C` numerically, the larger eigenvalue is `4.4209` with eigenvector pointing at `19.33°` — same angle. And scanning `l^T C l` brute-force over all directions, the maximum is `4.4209` at `19.33°` too, matching the larger eigenvalue, so `19.33°` is indeed where the variance is *greatest*, not least. So the closed form does land on the major axis, and the eigenvalue is exactly the variance captured along it. In 2D my "line of the cloud" *is* the major axis of the bivariate normal contour — a single, symmetric, well-defined direction, finally, instead of two regression lines. And there's a clean degenerate case that tells me when the problem has no answer: if the ellipsoid is a sphere — all correlations zero and all sigmas equal — then `C = sigma^2 I`, every direction gives the same `l^T C l = sigma^2`, and the `tan 2theta` formula has `0/0` on its right-hand side, undefined. So there's no unique best fit; the method correctly refuses to pick a direction when the data has no preferred one.

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

So the two routes — minimize perpendicular residual (the geometric, least-squares route) and maximize explained variance (the accounting route) — both reduce to an eigenproblem of the second-moment matrix. I want to be sure that's really the *same* eigenproblem and not just two that happen to look alike, because the geometric route diagonalized the covariance `C` and the accounting route diagonalized the correlation `R`. They are: `R` is `C` with rows and columns divided by the sigmas, which is exactly `C` computed on the standardized data `z_i = (x_i - xbar_i)/sigma_i`. So the two routes are the same operation — eigendecompose the second-moment matrix of the data — applied with two different choices of scaling on the variables, covariance for raw units and correlation for standardized ones. Whichever scaling I pick, the geometric and accounting derivations give the identical directions on that scaled data, because both extremize the same quadratic form `l^T C l` under `||l||=1`; the minimum gives the discarded normal, the maximum the kept direction. The first new variable is the top eigenvector of the (covariance or correlation) matrix; its eigenvalue is the variance it captures.

One detail I glossed: covariance matrix or correlation matrix? It depends on the units. If the `q` variables are in comparable, meaningful units, the natural metric is Euclidean in the raw coordinates and I diagonalize the covariance. If they're in arbitrary, incommensurable units — test "scores" with no shared scale — then a rescaling of one variable would arbitrarily squeeze or stretch the ellipsoid and move the principal axes, so I first standardize each variable to zero mean and unit variance, `z_i = (x_i - xbar_i)/sigma_i`, and diagonalize the correlation matrix instead. The method needs a *metric* — a notion of distance in the `q`-space — and the choice of covariance-vs-correlation is exactly the choice of that metric. There's no universal right answer; it's a modeling decision about what "equal importance" means. I'll keep the default as centering (the covariance route), which is what you want when the coordinates already share a scale, and note that standardizing first is the correlation route.

Now I need the *rest* of the components, not just the first. The recipe writes itself from the accounting view: once `gamma_1` is fixed, the next component should account for as much of the *residual* variance as possible — the variance left after `gamma_1`'s contribution is removed. The contribution of `gamma_1` to the cross-structure is the rank-one piece `a_i1 a_j1` (the outer product of the first loading vector), and the residual structure is what's left after I strip it out:
```
R' = R - a_1 a_1^T,
```
where the loading vector `a_1` is scaled so `a_1^T a_1 = k_1`. This is precisely the correlation/covariance matrix of the residuals `z_i - a_i1 gamma_1`, the part of the data uncorrelated with `gamma_1`. Then `gamma_2` is the top eigenvector of `R'`, `gamma_3` the top of `R''`, and so on — peel off one rank-one layer at a time. The reason this works is that the eigenvectors of a symmetric matrix are mutually orthogonal, so subtracting `k_1 v_1 v_1^T` should zero out the first eigenvalue while leaving every other eigenpair untouched, turning the previously-second eigenvector into the new top. The whole sequential procedure rests on this, so I check it on the same `3`-space covariance. The original eigenvalues, descending, are `(5.284, 0.790, 0.288)`. After forming `C' = C - 5.284 v_1 v_1^T` and diagonalizing, the eigenvalues come out `(0.790, 0.288, ~0)` — the first root is driven to zero (down to `-2e-15` roundoff) and the other two survive unchanged — and the top eigenvector of `C'` matches the original second eigenvector up to sign. So deflation does exactly what I need: it removes the captured direction and re-exposes the next one. So I don't even have to deflate in principle — the full eigen-decomposition hands me all the components at once, ordered by eigenvalue — but deflation is how I'd compute them one at a time if I only want the top few. Orthogonality among the loadings is automatic: `a_h^T a_m = delta_{hm} k_h`, columns orthogonal, lengths the square roots of the eigenvalues.

How much of the whole picture have I kept after `m` components? The total variance is the sum of all the eigenvalues — for the correlation matrix that's `q` (the trace, ones on the diagonal) — so the fraction of total variance carried by the `i`-th component is `k_i / (sum of all roots)`, and the cumulative fraction `(k_1 + ... + k_m)/(sum)` tells me how faithful an `m`-dimensional summary is. That's the natural stopping rule: keep enough components to capture the variance I care about, and the ones I drop are the small-eigenvalue directions where the cloud is thin — noise and redundancy.

Now, computing this. Solving `det(R - k I) = 0` explicitly — expanding the determinant into its characteristic polynomial `(-1)^q (k^q - q k^{q-1} + S_2 k^{q-2} - ...)` and rooting it, then back-substituting into the homogeneous system for each eigenvector — is brutal by hand for any real `q`. I want something cheaper that gives me the *largest* root and its eigenvector first, since those are the ones I keep, and lets me stop early. I can avoid the determinant by taking any trial direction `a` and hitting it with the matrix, `a' = R a`. Geometrically `R` stretches space most along its dominant principal axis, so `a'` is `a` rotated *toward* the top eigenvector. Iterate: `a -> R a -> R^2 a -> ...`. Expand the trial vector in the eigenbasis, `a = sum_j c_j v_j`; then `R^t a = sum_j c_j k_j^t v_j`, and since `k_1` is the largest root, `k_1^t` dominates every other `k_j^t` geometrically, so the iterate should align with `v_1` and the per-step scale factor should converge to `k_1`. Cranking a few steps on the `2x2` `C = [[4,1.2],[1.2,1]]` from before, starting deliberately off-axis at `a = (1, 0)` and reading off at each step the ratio of the new iterate to the old along the first coordinate (the common ratio that should converge to `k_1`): the ratios come out `4.000, 4.360, 4.413, 4.420, 4.4209, 4.4209` — climbing and settling — and the direction marches `16.70° -> 18.99° -> 19.28° -> 19.32° -> 19.33°`. The true top eigenvalue is `4.42094` and the major-axis angle was `19.33°`, so after six cheap matrix-vector products the iteration has both the root and the direction to four figures, with no determinant expanded. Repeat until the digits stop moving, normalize so `sum a_i^2 = k_1`, and I have the first component. The only way it fails is if the starting vector has zero component along `v_1`, an exceptional case I can avoid by changing the trial values. Then deflate — subtract `a_1 a_1^T` from the matrix — and run the same iteration on the reduced matrix for the second component, and so on. The running sum of the roots I've extracted, compared against the known total, tells me at every stage how much variance I've accounted for, so I know exactly when to stop. This power-iteration-plus-deflation is the hand-computable engine.

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
The SVD gives me the same `V` and the same ordering (singular values come sorted, so I don't even have to sort eigenvalues), but it never forms the ill-conditioned Gram matrix, so it's the numerically sound way to do the exact same linear algebra. On the `3`-space cloud the two agree: `S^2/(n-1)` from `svd(Xc)` is `(5.284, 0.790, 0.288)`, the same triple `eigh(C)` returned, and the leading right singular vector matches the top covariance eigenvector entrywise up to sign.

The projection onto the top `k` components is then `Xc @ V[:, :k]`, the coordinates of each centered point in the principal-axis frame. And there's a tiny but real reproducibility wrinkle: the sign of each singular vector is arbitrary — `U S V^T = (-U) S (-V)^T` is the same factorization — so two runs (or two LAPACK builds) can flip the sign of a component, flipping that coordinate's axis. To make the output deterministic I fix each right singular vector's sign by a fixed rule: force the largest-magnitude entry of that row to be positive. With a low-dimensional target like a 2D embedding, I just keep the first two columns. Note that the projected coordinates `Xc @ V[:, :k]` are exactly the first `k` columns of `U S` — `Xc V = U S V^T V = U S` — so I can read the embedding straight off the SVD without a separate matmul.

In code this is exactly the sequence just derived: center, full SVD of the centered matrix, sign-fix each component, keep the top right singular vectors as the directions, and return the top columns of `U S` as the same coordinates as `Xc @ Vt[:k].T`.

The two derivations — minimize perpendicular residual, maximize explained variance — landed on the same eigenproblem of the second-moment matrix by independent routes: centering removes the location nuisance, the eigenvectors give the ordered directions with a built-in stopping rule from the eigenvalues, and the SVD computes those same eigenvectors without forming the ill-conditioned Gram matrix.
