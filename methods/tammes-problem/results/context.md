# Context: the Tammes problem and spherical codes

## Research question

Place `N` points on the unit sphere `S^2` so that the minimum pairwise distance is as large as possible. Writing `psi(X) = min_{x != y} dist(x, y)` for the closest pair of a finite set `X` (with `dist` the geodesic angle, equivalently the chordal distance — the two orderings agree), the goal is

```
d_N = max_{|X| = N, X ⊂ S^2}  psi(X).
```

Equivalently: how should `N` congruent non-overlapping spherical caps be packed on a sphere so that their common radius is maximal — the botanist's question of how the exit pores distribute on a pollen grain. The same object is a *spherical code*: a set `X` with `<x, y> <= cos psi` for all distinct pairs.

The question splits into two halves of utterly different character. A **lower bound** `d_N >= delta` is a *construction*: exhibit `N` points that are pairwise `>= delta` apart and you are done — a picture suffices. The **upper bound** `d_N <= delta`, together with **uniqueness** of the optimal arrangement up to isometry, is the hard half: it must rule out a *continuum* of configurations at once, certifying that no arrangement of `N` points does better. "Solving" the Tammes problem for a given `N` means closing this gap exactly and proving uniqueness — not reporting a numerically optimized arrangement. The pollen-grain values that have been pinned exactly are `N = 3, 4, 6, 12` (Fejes Tóth 1943), `N = 5, 7, 8, 9` (Schütte–van der Waerden 1951), `N = 10, 11` (Danzer 1963), and `N = 24` (Robinson 1961); `N = 13` and `N = 14` stood open for decades and are the natural test of whether a method is genuinely principled.


## Background

**The packing / kissing connection.** If `N` unit spheres kiss a central unit sphere in `R^3`, their contact points lie on `S^2` with pairwise angular separation `>= 60` degrees, so `d_N >= 60°` is impossible once `N` exceeds the kissing number `k(3) = 12`. Hence `d_13 < 60°` for free, and `N = 13` is exactly the first case where the kissing constraint bites — which is why it sat next to the Newton–Gregory thirteen-spheres dispute and the Kepler problem.

**Global analytic upper bounds.** Two families of dimension-independent bounds existed. The first is geometric-combinatorial. A set of `N` points triangulated on the sphere has, by Euler, `2N - 4` triangular faces; if every edge is `>= d` then every face contains a spherical triangle of side `>= d`, whose area (angular excess) is at least that of the *equilateral* one. The angle of an equilateral spherical triangle of side `d` is

```
alpha(d) = arccos( cos d / (1 + cos d) ),
```

with area `3 alpha(d) - pi`. Summing, `(2N - 4)(3 alpha(d) - pi) <= 4 pi`, and inverting gives the Fejes Tóth bound

```
d_N <= arccos( c_N / (1 - c_N) ),   c_N = cos( pi N / (3N - 6) ),
```

which is sharp exactly when the sphere tiles into equilateral triangles (`N = 3, 4, 6, 12`) and slack otherwise — e.g. `58.6809°` at `N = 14`, and `60.92°` at `N = 13`, weaker there than the kissing-number fact `d_13 < 60°`.

The second family is the linear-programming bound for spherical codes. Functions on `S^{n-1}` decompose by spherical-harmonic degree; the one-variable shadow is the **Gegenbauer (ultraspherical) polynomials** `P_k^{(n)}`, defined by `P_0 = 1`, `P_1 = t`, the three-term recurrence

```
(k + n - 2) P_{k+1}^{(n)}(t) = (2k + n - 2) t P_k^{(n)}(t) - k P_{k-1}^{(n)}(t),
```

normalized `P_k^{(n)}(1) = 1` (for `n = 3` these are the Legendre polynomials). The load-bearing fact is the **addition theorem**: with `{v_{k,j}}` an orthonormal basis of the degree-`k` harmonics of dimension `r_k`,

```
P_k^{(n)}(<x, y>) = (1/r_k) sum_j v_{k,j}(x) v_{k,j}(y),
```

a positive-definite kernel. Schoenberg (1942) proved `f(cos theta)` is positive-definite on `S^{n-1}` iff `f = sum_k f_k P_k^{(n)}` with all `f_k >= 0`. From the resulting identity, for any finite code `C`,

```
|C| f(1) + sum_{x != y} f(<x,y>) = |C|^2 f_0 + sum_{k>=1} (f_k / r_k) sum_j ( sum_{x in C} v_{k,j}(x) )^2,
```

so if `f_0 > 0`, `f_k >= 0` for `k >= 1`, and `f(t) <= 0` on `[-1, cos psi]`, the right side is `>= |C|^2 f_0` and the left side's off-diagonal sum is `<= 0`, giving `|C| <= f(1)/f_0`. This bounds the *cardinality* of a code at a fixed angle. It is tight for the kissing problem only in dimensions `8` and `24`; on `S^2` it brackets `d_13 < 58.5°` and `d_14 < 56.58°` (the latter via the semidefinite strengthening of Bachoc–Vallentin).

**The local structure of an optimum.** The motivating empirical observation, going back to Schütte–van der Waerden and Danzer, is that an optimal arrangement is *jammed*: at the maximum, the configuration is rigid — there is no way to nudge any subset of points to strictly increase the closest distance, and the pairs sitting exactly at the minimum distance are the ones holding it in place. Slack pairs (distance strictly greater than the minimum) impose no constraint on local motion. This jamming was read off the small-`N` optima long before any general theory of it was available.

## Baselines

**Spherical-trigonometry case analysis (Schütte–van der Waerden 1951–53; Danzer 1963).** The classical route to small-`N` optima: study the irreducible taut-pair graph, enumerate its possible combinatorial types *by hand*, and use spherical trigonometry plus area arguments to discard all but the optimum. This is rigorous and is how `N <= 11` were settled.

**Fejes Tóth area bound (1943).** The `(2N-4)(3 alpha(d) - pi) <= 4 pi` bound above. General and clean, sharp exactly when the sphere tiles into equilateral triangles (`N = 3, 4, 6, 12`).

**Delsarte LP bound and its SDP strengthening (Delsarte–Goethals–Seidel 1977; Kabatiansky–Levenshtein 1978; Bachoc–Vallentin 2008).** The cardinality bound `|C| <= f(1)/f_0` and its three-point semidefinite refinement. Uniform and powerful — it solved the kissing number in dimensions `8` and `24`, and Musin's modification (relaxing the sign constraint near `t = 1` and counting points in a cap) closed `k(3) = 12` and `k(4) = 24`. For the Tammes problem it gives `d_13 < 58.5°` and `d_14 < 56.58°`.

**Connelly's rigidity / stress-matrix machinery (Connelly 2005).** Tensegrity theory supplies a certificate language for first-order stationarity of constrained frameworks, in terms of equilibrium stresses on the active edges, developed for bar-and-cable frameworks.

## Evaluation settings

The yardstick is the table of `d_N` for small `N`, comparing the best construction (lower bound, from Sloane's spherical-code tables) against a proof of optimality (upper bound + uniqueness). The decisive open cases are `N = 13` and `N = 14`: long-conjectured optimal arrangements `P_13` (with `psi(P_13) ≈ 57.1367°`) and `P_14` (`psi(P_14) ≈ 55.67057°`) exist as constructions, bracketed above by `58.5°` and `56.58°` respectively. Angles are measured geodesically on `S^2`; the equivalent inner-product form uses `<x, y> <= cos psi`.

## Code framework

Pre-existing primitives: array arithmetic, polynomial arithmetic and root-building, a linear-program solver (`scipy.optimize.linprog` / GLPK), interval arithmetic, subprocess access to an isomorph-free planar-graph generator, and basic spherical trigonometry. The empty slots are to be filled in.

```python
import subprocess
import numpy as np
from numpy.polynomial import polynomial as Pp
from scipy.optimize import linprog

def _pad(poly, size):
    pass

def _row(coeffs, size):
    pass

def _poly_max_on_interval(poly_coeffs, lo, hi):
    pass

def area_upper_bound(num_points):
    """Return a geometric upper bound on the optimal angle."""
    pass

def gegenbauer_basis(dimension, max_degree):
    """Return the zonal positive-definite polynomial basis through max_degree."""
    pass

def code_cardinality_bound(poly_coeffs, dimension, inner_product_ceiling,
                           offdiag_certifier=None):
    """Turn a one-variable polynomial certificate into a code-size bound."""
    pass

def triangle_angle(side_length):
    """Return the face angle determined by an equilateral spherical triangle."""
    pass

def opposite_angle(angle, side_length):
    """Return the paired angle determined by the spherical quadrilateral relation."""
    pass

def candidate_stream(num_points, generator="plantri", options=("-a",)):
    """Yield isomorph-free candidates from the external generator."""
    pass

def lp_empty(num_variables, bounds, equalities=(), inequalities=()):
    """Return True when the supplied linear feasibility problem is infeasible."""
    pass

# Further empty slots to be filled in as the analysis dictates.
```

The content to be supplied is what these slots must compute.
