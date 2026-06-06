# Context: the Tammes problem and spherical codes

## Research question

Place `N` points on the unit sphere `S^2` so that the minimum pairwise distance is as large as possible. Writing `psi(X) = min_{x != y} dist(x, y)` for the closest pair of a finite set `X` (with `dist` the geodesic angle, equivalently the chordal distance — the two orderings agree), the goal is

```
d_N = max_{|X| = N, X ⊂ S^2}  psi(X).
```

Equivalently: how should `N` congruent non-overlapping spherical caps be packed on a sphere so that their common radius is maximal — the botanist's question of how the exit pores distribute on a pollen grain. The same object is a *spherical code*: a set `X` with `<x, y> <= cos psi` for all distinct pairs.

The question splits into two halves of utterly different character. A **lower bound** `d_N >= delta` is a *construction*: exhibit `N` points that are pairwise `>= delta` apart and you are done — a picture suffices. The **upper bound** `d_N <= delta`, together with **uniqueness** of the optimal arrangement up to isometry, is the hard half: it must rule out a *continuum* of configurations at once, certifying that no arrangement of `N` points does better. "Solving" the Tammes problem for a given `N` means closing this gap exactly and proving uniqueness — not reporting a numerically optimized arrangement. The pollen-grain values that have been pinned exactly are `N = 3, 4, 6, 12` (Fejes Tóth 1943), `N = 5, 7, 8, 9` (Schütte–van der Waerden 1951), `N = 10, 11` (Danzer 1963), and `N = 24` (Robinson 1961); `N = 13` and `N = 14` stood open for decades and are the natural test of whether a method is genuinely principled.

What a satisfying solution must produce: a *finite* certificate that the continuous optimization over the `2N`-dimensional configuration manifold attains its maximum exactly at one arrangement.

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

so if `f_0 > 0`, `f_k >= 0` for `k >= 1`, and `f(t) <= 0` on `[-1, cos psi]`, the right side is `>= |C|^2 f_0` and the left side's off-diagonal sum is `<= 0`, giving `|C| <= f(1)/f_0`. This bounds the *cardinality* of a code at a fixed angle. It is tight for the kissing problem only in dimensions `8` and `24`; on `S^2` it brackets `d_13 < 58.5°` and `d_14 < 56.58°` (the latter via the semidefinite strengthening of Bachoc–Vallentin) but produces a numerical gap, not the exact `d_N` nor a uniqueness statement.

**The local structure of an optimum.** The motivating empirical observation, going back to Schütte–van der Waerden and Danzer, is that an optimal arrangement is *jammed*: at the maximum, the subgraph of pairs that are exactly at the minimum distance is rigid — there is no way to nudge any subset of points to strictly increase the closest distance. Two elementary local moves capture every way the configuration could improve: a **shift**, sliding a single point off the others to gain room; and **Danzer's flip**, taking a point `x` with two taut neighbours `y, z` and reflecting it across the great circle `yz` to land further from the rest. An arrangement admitting neither move is *irreducible*. Danzer further observed, by elementary spherical geometry, that in an irreducible arrangement the taut-pair graph is planar, every vertex has degree `0, 3, 4,` or `5`, and (because the angle exceeds roughly `55°`, so `2pi/d < 7`) every face is a convex equilateral spherical polygon with at most six sides, with isolated points only inside hexagons. These are properties of any optimum, known before any particular configuration is named.

## Baselines

**Spherical-trigonometry case analysis (Schütte–van der Waerden 1951–53; Danzer 1963).** The classical route to small-`N` optima: study the irreducible taut-pair graph, enumerate its possible combinatorial types *by hand*, and use spherical trigonometry plus area arguments to discard all but the optimum. This is rigorous and is how `N <= 11` were settled, but it is a heavy, ad hoc, case-by-case enumeration; the number of admissible graphs grows explosively and by `N = 13` (tens of millions of candidate graphs) hand enumeration is hopeless. It also leaves uniqueness delicate.

**Fejes Tóth area bound (1943).** The `(2N-4)(3 alpha(d) - pi) <= 4 pi` bound above. General and clean, but sharp only at `N = 3, 4, 6, 12`; at the open cases it is far too loose (it does not even beat `60°` at `N = 13`).

**Delsarte LP bound and its SDP strengthening (Delsarte–Goethals–Seidel 1977; Kabatiansky–Levenshtein 1978; Bachoc–Vallentin 2008).** The cardinality bound `|C| <= f(1)/f_0` and its three-point semidefinite refinement. Uniform and powerful — it solved the kissing number in dimensions `8` and `24`, and Musin's modification (relaxing the sign constraint near `t = 1` and counting points in a cap) closed `k(3) = 12` and `k(4) = 24`. But for the Tammes problem it bounds the wrong quantity: it caps how many points fit at a fixed angle, yielding `d_13 < 58.5°`, `d_14 < 56.58°` — a numerical bracket above the true value, with no exact equality and no uniqueness.

**Connelly's rigidity / stress-matrix machinery (Connelly 2005).** Tensegrity theory supplies a certificate language for first-order stationarity: at a constrained maximum, the active edges must support a nonnegative equilibrium stress. Developed for bar-and-cable frameworks, not yet aimed at packings, it gives a way to falsify a candidate taut graph when no such stress can exist.

The gap all of these leave: the analytic bounds give brackets, not exact values; the hand enumeration does not scale past `N = 11`. Nothing turns the continuous optimization at `N = 13, 14` into a *finite, checkable* proof of the exact optimum and its uniqueness.

## Evaluation settings

The yardstick is the table of `d_N` for small `N`, comparing the best construction (lower bound, from Sloane's spherical-code tables) against a proof of optimality (upper bound + uniqueness). The decisive cases are `N = 13` and `N = 14`: long-conjectured optimal arrangements `P_13` (with `psi(P_13) ≈ 57.1367°`) and `P_14` (`psi(P_14) ≈ 55.67057°`) exist as constructions, bracketed above by `58.5°` and `56.58°` respectively; a method is judged principled only if it collapses each bracket to the exact value *and* certifies uniqueness up to isometry, with a proof that is finite and machine-checkable. Angles are measured geodesically on `S^2`; the equivalent inner-product form uses `<x, y> <= cos psi`.

## Code framework

Pre-existing primitives: array arithmetic, polynomial arithmetic and root-building, a linear-program solver (`scipy.optimize.linprog` / GLPK), interval arithmetic, subprocess access to an isomorph-free planar-graph generator, and basic spherical trigonometry. The empty slots are a global-bound calculator, a planar-candidate stream, a linear feasibility test for certified outer cells, and an equilibrium certificate for hard survivors.

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

def planar_candidate_stream(num_points, generator="plantri", options=("-a",)):
    """Yield isomorph-free planar candidates from an external generator."""
    pass

def passes_combinatorial_screen(degrees, face_sizes):
    """Apply local degree and face-size necessary conditions."""
    pass

def angle_sum_equalities(vertex_incidence, num_variables, angle_offset=0):
    """Build full-turn linear equalities around vertices."""
    pass

def lp_empty(num_variables, bounds, equalities=(), inequalities=()):
    """Return True when the supplied linear feasibility problem is infeasible."""
    pass

def eliminate_candidate_by_lp(cells):
    """Return True only when every certified outer cell is LP-infeasible."""
    pass

def stress_infeasible(edge_list, direction_boxes, num_points):
    """Return True when no normalized nonnegative equilibrium stress exists."""
    pass
```

The two deciders are the whole content: what algebraic conditions make a polynomial a valid code-size certificate, and what linear feasibility question certifies that a candidate planar type cannot host the extremal arrangement.
