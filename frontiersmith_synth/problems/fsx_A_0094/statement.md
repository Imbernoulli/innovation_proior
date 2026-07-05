# Sparse Telescope Array: Baseline Diversity in a Triangular Reserve

## Problem
You are laying out a radio-interferometer array. A triangular land reserve is modelled as the
**unit right triangle**

```
T = conv{ (0,0), (1,0), (0,1) }   (a point (x,y) is in T iff  x >= 0,  y >= 0,  x + y <= 1 )
```

You must place exactly `N` telescope **stations** at coordinates inside `T`. In aperture
synthesis the imaging quality of any three stations is governed by the **area of the triangle**
they form (a fat triangle samples the Fourier plane in three well-separated directions; a thin,
near-collinear triple is nearly redundant). The array is only as good as its **worst** triple,
so mission planners score a layout by the **minimum triangle area over all triples of stations**.

Output a placement of the `N` stations that makes this worst-case triangle area **as large as
possible**. (This is the Heilbronn-triangle configuration problem on the triangle: the optimum is
not known in closed form for these `N`, and many different layouts are viable.)

## Input (stdin)
One line with a single integer `N`, the number of stations to place.
```
N
```

## Output (stdout)
Exactly `N` lines, each with two floats `x y`, the coordinates of one station:
```
x_1 y_1
x_2 y_2
...
x_N y_N
```
Stations may appear in any order. Output exactly `2 * N` numbers total.

## Feasibility
All checks use tolerance `tol = 1e-6`:
- Exactly `2 * N` numbers are present and every number is finite (no `nan`/`inf`).
- Containment: each station satisfies `x >= -tol`, `y >= -tol`, `x + y <= 1 + tol`.

Any violation scores `Ratio: 0.0`. (Coincident or collinear stations are *allowed* but yield a
zero-area triple, driving the objective — and hence the score — to `0`.)

## Objective (maximize)
For a layout `P = {p_1, ..., p_N}`,
```
F(P) = min over all triples {i<j<k} of  area( p_i, p_j, p_k )
```
where `area` is the exact Euclidean triangle area `|(p_j - p_i) x (p_k - p_i)| / 2`.

## Scoring
Let `B` be the checker's internal baseline: `N` stations equally spaced on a small circle of
radius `R = r_in / sqrt(3)` about the incenter of `T` (`r_in = (2 - sqrt(2)) / 2`). With your
feasible objective `F`,
```
sc    = min(1000, 100 * F / max(1e-9, B))
Ratio = sc / 1000
```
so reproducing the baseline ring scores about `0.1`, and a layout ten times better caps at `1.0`.

## Constraints
`6 <= N <= 15`. Runs comfortably within the time limit for these sizes.

## Example
For `N = 3`, three stations at the triangle's corners `(0,0),(1,0),(0,1)` form a single triangle
of area `0.5`, so `F = 0.5`. The baseline ring of 3 points (radius `R ≈ 0.169`) forms an
equilateral triangle of area `B ≈ 0.037`, giving `Ratio = min(1000, 100 * 0.5 / 0.037)/1000 = 1.0`
(capped). This tiny illustrative case is easy; the graded ladder uses `N` from 6 to 15, where the
worst triple is forced to be thin and no layout is obviously optimal.
