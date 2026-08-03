# Cryo Qubit Junction Layout: Sliver-Free Wire-Bond Pads

## Problem
A dilution-refrigerator carries a **triangular** cryostat plate onto which `N`
wire-bond junction pads must be placed. During calibration the control
electronics pick **three** pads and use the triangle those pads span as a local
reference frame to solve for wire-crosstalk coefficients. If the three chosen
pads are nearly collinear (a thin sliver triangle) the reference frame is
ill-conditioned and the calibration is garbage. To keep **every** possible
triple of pads usable, place the pads so that the **smallest-area triangle**
spanned by any three of them is **as large as possible**.

This is a Heilbronn-type extremal configuration problem on the unit triangle: no
closed-form optimum is known for general `N`, several very different layouts are
competitive, and small local moves keep paying off, so the search is genuinely
open-ended.

## Input (stdin)
```
N
```
`N` = number of junction pads. The plate is the fixed unit right triangle with
corners `(0,0)`, `(1,0)`, `(0,1)`; only `N` varies across tests.

## Output (stdout)
Exactly `N` lines, each `x y` (floats): the coordinates of one pad.

## Feasibility
With absolute tolerance `tol = 1e-6`:
- Exactly `N` points must be provided; every coordinate must be a **finite**
  number (no `nan`/`inf`).
- Every pad must lie inside the plate: `x >= -tol`, `y >= -tol`, and
  `x + y <= 1 + tol`.

Any violation scores `Ratio: 0.0`. Coincident or collinear pads are allowed but
span a zero-area triangle, which crushes the objective.

## Objective (maximize)
```
F = min over all C(N,3) triples of the triangle area they span
```
computed exactly as `0.5 * |cross product|`.

## Scoring
Let `B` be the checker's internal baseline: `N` pads equally spaced on a circle
of radius `0.15` centered at the plate centroid `(1/3, 1/3)` (a regular `N`-gon
inscribed in the plate, whose minimum-area triangle is a thin 3-consecutive-vertex
sliver). With `F` your feasible value,
```
sc    = min(1000, 100 * F / max(1e-9, B))
Ratio = sc / 1000
```
so reproducing the ring baseline scores about `0.1`, and a layout ten times
better caps at `1.0`.

## Constraints
`8 <= N <= 17`. The checker runs in `O(N^3)`; every size finishes far under the
time limit.

## Example
Suppose `N = 3`. Placing pads at the three corners `(0,0), (1,0), (0,1)` spans a
single triangle of area `1/2`, so `F = 0.5`. If the ring baseline `B` for `N=3`
were, say, `0.049`, then
`Ratio = min(1000, 100*0.5/0.049)/1000 = 1.0` (capped).
(Three corners is illustrative only — it is trivial for `N=3` and not
competitive for the larger `N` actually tested.)
