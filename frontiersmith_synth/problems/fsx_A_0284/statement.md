# Summit Ridge Resort: Well-Spread Lift Tower Layout

## Problem
Summit Ridge is building `N` chairlift towers across a square alpine plot. When
avalanche-control crews triangulate a snow-load reading, they pick **three**
towers and use the triangle those towers span as a reference frame. If the three
chosen towers are nearly collinear (a thin sliver triangle) the reading is
garbage. To keep **every** possible triple of towers usable, place the towers so
that the **smallest-area triangle** spanned by any three of them is **as large
as possible**.

This is a Heilbronn-type extremal configuration problem on the unit square: no
closed-form optimum is known for general `N`, several very different layouts are
competitive, and small local moves keep paying off, so the search is genuinely
open-ended.

## Input (stdin)
```
N
```
`N` = number of lift towers. The plot is the fixed unit square with corners
`(0,0)`, `(1,0)`, `(1,1)`, `(0,1)`; only `N` varies across tests.

## Output (stdout)
Exactly `N` lines, each `x y` (floats): the coordinates of one tower.

## Feasibility
With absolute tolerance `tol = 1e-6`:
- Exactly `N` points must be provided; every coordinate must be a **finite**
  number (no `nan`/`inf`).
- Every tower must lie inside the plot: `-tol <= x <= 1+tol` and
  `-tol <= y <= 1+tol`.

Any violation scores `Ratio: 0.0`. Coincident or collinear towers are allowed
but span a zero-area triangle, which crushes the objective.

## Objective (maximize)
```
F = min over all C(N,3) triples of the triangle area they span
```
computed exactly as `0.5 * |cross product|`.

## Scoring
Let `B` be the checker's internal baseline: `N` towers equally spaced on a circle
of radius `0.20` centered at the plot's middle `(0.5, 0.5)` (a regular `N`-gon,
whose minimum-area triangle is a thin 3-consecutive-vertex sliver). With `F` your
feasible value,
```
sc    = min(1000, 100 * F / max(1e-9, B))
Ratio = sc / 1000
```
so reproducing the ring baseline scores about `0.1`, and a layout ten times
better caps at `1.0`.

## Constraints
`7 <= N <= 16`. The checker runs in `O(N^3)`; every size finishes far under the
time limit.

## Example
Suppose `N = 4`. Placing towers at the four corners `(0,0), (1,0), (1,1), (0,1)`
spans four triangles, each of area `1/2`, so `F = 0.5`. If the ring baseline `B`
for `N=4` were, say, `0.29`, then
`Ratio = min(1000, 100*0.5/0.29)/1000 ≈ 0.172`.
(Four corners is illustrative only, not competitive for larger `N`.)
