# Glacier Sensor Net: Well-Spread Seismic Beacon Layout

## Problem
A glacier survey team must drop `N` seismic beacons onto a triangular study
region. When three beacons are used together to triangulate a micro-quake, the
*resolution* of that reading degrades badly if the three beacons are nearly
collinear (the spanned triangle is a thin sliver). To keep **every** possible
triangulation trustworthy, you want to place the beacons so that the **smallest
triangle** spanned by any three of them is **as large as possible**.

This is a Heilbronn-type extremal configuration problem posed on a triangle: no
closed-form optimum is known for general `N`, and many different layouts are
worth trying.

## Input (stdin)
```
N
ax ay
bx by
cx cy
```
`N` = number of beacons. The next three lines give the corners `A, B, C` of the
survey triangle (listed counter-clockwise). In this task the region is the fixed
right triangle `A=(0,0)`, `B=(1,0)`, `C=(0,1)`; only `N` changes across tests.

## Output (stdout)
Exactly `N` lines, each `x y` (floats): the coordinates of one beacon.

## Feasibility
With relative barycentric tolerance `tol = 1e-6`:
- Exactly `N` points must be provided; all coordinates finite.
- Every beacon must lie inside the survey triangle (barycentric coordinates all
  `>= -tol`).

Any violation scores `Ratio: 0.0`. Coincident or collinear beacons are allowed
but yield a zero-area triangle, so they crush the objective.

## Objective (maximize)
`F = min over all C(N,3) triples of the triangle area they span`,
computed exactly as `0.5 * |cross product|`.

## Scoring
Let `B` be the checker's internal baseline: `N` beacons equally spaced on the
survey triangle's **incircle** (a regular `N`-gon), whose minimum triangle area
is dominated by thin 3-consecutive-vertex slivers. With `F` your feasible value,
```
sc    = min(1000, 100 * F / max(1e-9, B))
Ratio = sc / 1000
```
so reproducing the incircle baseline scores about `0.1`, and a layout ten times
better caps at `1.0`.

## Constraints
`6 <= N <= 21`. The checker runs in `O(N^3)`; all sizes finish well under the
time limit.

## Example
Suppose `N = 4`. Placing beacons at the three corners and the centroid,
`(0,0), (1,0), (0,1), (1/3,1/3)`, the four spanned triangles each have area
`1/6 ≈ 0.16667`, so `F = 0.16667`. If the incircle baseline `B` for `N=4` is,
say, `0.061`, then `Ratio = min(1000, 100*0.16667/0.061)/1000 ≈ 0.273`.
(Corner-and-centroid is illustrative only, not optimal for larger `N`.)
