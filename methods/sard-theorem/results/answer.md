# Sard's Theorem

## Statement

Let `f : U -> R^m` be smooth, with `U` open in `R^n`. A point `x in U` is critical if `rank Df_x < m`; a value `y in R^m` is critical if `y = f(x)` for some critical point `x`. The set of critical values has Lebesgue measure zero in `R^m`.

The finite-differentiability form is sharper: the same conclusion holds for `C^r` maps when `r > max(n-m, 0)`. The smooth statement is the clean version used for the proof artifact below.

For a smooth map between smooth manifolds, the critical values have measure zero in the target manifold, measured in coordinate charts.

## Core Mechanism

Rank deficiency squeezes volume in the target. Near a critical point, the derivative sends the domain into a proper subspace of the target. Taylor expansion turns that first-order rank loss into a quantitative estimate on small boxes. A covering argument then sums the local squeezed images and shows their total target volume can be made arbitrarily small.

The proof has to avoid the false claim that critical points are small. They need not be. The theorem measures the image of the critical set.

## Proof Artifact

It is enough to prove the Euclidean statement on compact subsets of coordinate boxes, then take countable unions.

Work by induction on the source dimension `n`. The case `n=0` is immediate. Let `C` be the critical set of a smooth `f : U subset R^n -> R^m`.

First handle points where `0 < rank Df_x = r < m`. After reordering coordinates, choose a nonzero `r x r` minor. Locally the map

`x -> (f_1(x), ..., f_r(x), x_{r+1}, ..., x_n)`

is a diffeomorphism. In the resulting coordinates,

`f(u,v) = (u, G(u,v))`,

with `u in R^r`, `v in R^{n-r}`, and `G(u,v) in R^{m-r}`. The derivative has block form

`Df = [[I, 0], [D_u G, D_v G]]`,

so `f` is critical exactly when `D_v G(u,v)` has rank `< m-r`. For each fixed `u`, the bad target values of `G_u(v)=G(u,v)` have zero `(m-r)`-dimensional measure by the induction hypothesis. Fubini then gives zero `m`-dimensional measure for the corresponding critical values of `f` in the local product coordinates.

It remains to handle the rank-zero part. Let `K_j` be the set of points where all positive-order partial derivatives of all components of `f` through order `j` vanish. For points in `K_j \ K_{j+1}`, some derivative of order `j+1` is nonzero; equivalently, some derivative of order `j` has nonzero gradient. Since points of `K_j` make that derivative zero, the implicit function theorem places `K_j` locally inside a smooth hypersurface. Parametrizing that hypersurface reduces the source dimension by one, so the induction hypothesis shows the image of this piece has measure zero.

The residual set is `K_k` for a large `k`. Choose `k` with `k m > n`. On a compact cube, Taylor's theorem with all derivatives of order `< k` vanishing on `K_k` gives the following uniform estimate: for every small subcube `Q` of side `delta` meeting `K_k`, choose `x_Q in Q cap K_k`; then

`f(Q cap K_k)` is contained in a target ball of radius `epsilon(delta) delta^k`,

where `epsilon(delta) -> 0` as `delta -> 0`. Covering the compact cube by `O(delta^{-n})` such subcubes, the total `m`-dimensional volume of the target balls is bounded by

`C delta^{-n} (epsilon(delta) delta^k)^m`

`= C epsilon(delta)^m delta^{k m - n}`.

Because `k m > n`, this tends to zero. Hence `f(K_k)` has measure zero.

The critical set is covered by the positive-rank pieces, the hypersurface pieces `K_j \ K_{j+1}`, and the deeply flat piece `K_k`. Each contributes a measure-zero set of critical values. Countable localization over the domain gives the full Euclidean theorem.

For manifolds, take countably many source charts and target charts. In each chart the Euclidean theorem applies, and countable unions preserve measure zero. This proves the manifold statement.

## Consequences

Almost every value of a smooth map is regular. If `y` is regular, then each point of `f^{-1}(y)` has surjective derivative, so the submersion theorem makes `f^{-1}(y)` a smooth manifold of dimension `n-m` when nonempty. In transversality arguments, the same idea appears after building an auxiliary map whose critical values are precisely the bad parameters or nonregular intersections; the theorem makes those failures negligible.
