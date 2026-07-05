# Coral Reef Survey: Sliver-Free Station Placement in a Reef Plot

## Problem
A marine biology team must anchor `N` monitoring stations onto a square reef
plot. When three stations are combined to triangulate the position of a tagged
fish (or the epicentre of a coral-bleaching event), the *reliability* of that
reading collapses if the three stations are nearly collinear — the triangle they
span becomes a razor-thin sliver. To keep **every** possible three-station
triangulation trustworthy, place the stations so that the **smallest triangle**
spanned by any three of them is **as large as possible**.

This is a Heilbronn-type extremal configuration problem posed on a square: no
closed-form optimum is known for general `N`, and many different layouts are
worth trying.

## Input (stdin)
```
N
ax ay
bx by
cx cy
dx dy
```
`N` = number of stations. The next four lines give the corners `A, B, C, D` of
the reef plot (listed counter-clockwise). In this task the plot is the fixed
unit square `A=(0,0)`, `B=(1,0)`, `C=(1,1)`, `D=(0,1)`; only `N` changes across
tests.

## Output (stdout)
Exactly `N` lines, each `x y` (floats): the coordinates of one station. (Extra
whitespace/newlines are fine as long as exactly `2N` numeric tokens are printed.)

## Feasibility
With tolerance `tol = 1e-6`:
- Exactly `N` stations (i.e. `2N` numeric tokens) must be provided; all
  coordinates finite.
- Every station must lie inside the reef plot: `-tol <= x <= 1+tol` and
  `-tol <= y <= 1+tol`.

Any violation scores `Ratio: 0.0`. Coincident or collinear stations are allowed
but yield a zero-area triangle, which crushes the objective.

## Objective (maximize)
`F = min over all C(N,3) triples of the triangle area they span`,
computed exactly as `0.5 * |cross product|`.

## Scoring
Let `B` be the checker's internal baseline: `N` stations equally spaced on the
reef plot's **incircle** (a regular `N`-gon, centre `(0.5,0.5)`, radius `0.5`),
whose minimum triangle area is dominated by thin 3-consecutive-vertex slivers.
With `F` your feasible value,
```
sc    = min(1000, 100 * F / max(1e-9, B))
Ratio = sc / 1000
```
so reproducing the incircle baseline scores about `0.1`, and a layout ten times
better caps at `1.0`.

## Constraints
`6 <= N <= 15`. The checker runs in `O(N^3)`; all sizes finish well under the
time limit.

## Example
Suppose `N = 4`. Placing stations at the four plot corners
`(0,0), (1,0), (1,1), (0,1)` gives four spanned triangles each of area `1/2`, so
`F = 0.5`. If the incircle baseline `B` for `N=4` (a square inscribed in the
incircle) is `0.25`, then `Ratio = min(1000, 100*0.5/0.25)/1000 = 0.2`.
(Four-corners is illustrative only, not optimal for larger `N`.)
