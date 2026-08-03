# Pandemic Contact Net: Spread-Out Superspreader-Safe Layout

## Problem
A public-health team is planning the fixed floor positions of `N` people in a
square gathering hall for a monitored event. Contact tracers model a *risky
alignment* as any three attendees who stand nearly in a straight line: a
line-of-three lets a single exhaled plume reach all three at once, so the
skinnier the triangle those three span, the more dangerous that trio is. The
overall risk of the whole layout is set by its **worst** (thinnest) trio.

To make the contact net as safe as possible you must place the `N` people so
that the **smallest triangle** spanned by any three of them is **as large as
possible**.

This is a Heilbronn-type extremal point configuration posed on the unit square.
No closed-form optimum is known for the sizes used here, the objective is highly
non-convex, and many different layouts (rings, filled grids, corner-loaded
lattices, locally polished configurations) are worth trying.

## Input (stdin)
```
N
x0 y0
x1 y1
x2 y2
x3 y3
```
`N` = number of attendees. The next four lines give the corners of the hall in
counter-clockwise order. In this task the hall is the fixed unit square with
corners `(0,0)`, `(1,0)`, `(1,1)`, `(0,1)`; only `N` changes across tests.

## Output (stdout)
Exactly `N` lines, each `x y` (floats): the floor position of one attendee.

## Feasibility
With absolute tolerance `tol = 1e-6`:
- Exactly `N` positions must be provided; all coordinates finite.
- Every attendee must lie inside the hall: `-tol <= x <= 1+tol` and
  `-tol <= y <= 1+tol`.

Any violation scores `Ratio: 0.0`. Coincident or collinear attendees are
allowed but span a zero-area triangle, so they crush the objective.

## Objective (maximize)
`F = min over all C(N,3) triples of the triangle area they span`,
computed exactly as `0.5 * |cross product|`.

## Scoring
Let `B` be the checker's internal baseline: `N` attendees equally spaced on a
concentric ring of radius `0.3` centred at `(0.5, 0.5)` (a regular `N`-gon whose
minimum triangle area is dominated by thin three-consecutive-vertex slivers).
With `F` your feasible value,
```
sc    = min(1000, 100 * F / max(1e-9, B))
Ratio = sc / 1000
```
so reproducing the ring baseline scores about `0.1`, and a layout ten times
better caps at `1.0`.

## Constraints
`12 <= N <= 30`. The checker runs in `O(N^3)`; all sizes finish well under the
time limit.

## Example
Suppose `N = 4`. Placing the four attendees at the hall corners
`(0,0), (1,0), (1,1), (0,1)`, every one of the four spanned triangles has area
`0.5`, so `F = 0.5`. If the ring baseline `B` for `N = 4` is, say, `0.18`, then
`Ratio = min(1000, 100*0.5/0.18)/1000 ≈ 0.278`. (The four-corner layout is
illustrative only and is not optimal for the larger `N` used in the tests.)
