# Sourdough Sampling Plan: Low-Discrepancy Recipe Coverage

## Problem
A regional bakery is tuning a sourdough line. Every trial recipe is a point in the
unit square of two normalized dials: `x` = flour fraction and `y` = proof time.
The R&D team can afford only `M` trial bakes, so the trials must **cover the recipe
square as evenly as possible** — statistically, this means the sample set should have
the smallest possible **star discrepancy**.

Some of the trials are non-negotiable: `K` **signature recipes** (anchor points) are
already committed and must appear in the plan. You choose the remaining `M - K` points
freely to minimize the star discrepancy of the whole set of `M` points.

## Input (stdin)
```
d M K
ax_1 ay_1
...
ax_K ay_K
```
- `d = 2` (the recipe space is always the unit square).
- `M` = total number of trial points to output.
- `K` = number of fixed signature recipes; each `(ax_i, ay_i)` lies in `(0,1)^2`.

## Output (stdout)
Exactly `M` lines (or `M` whitespace-separated coordinate pairs), each
```
x y
```
with `0 <= x <= 1` and `0 <= y <= 1`. The set MUST contain every anchor point
(matched within `1e-4` per coordinate).

## Feasibility
The output is rejected (score 0) if: it does not contain exactly `M` points, any
coordinate lies outside `[0,1]`, or any of the `K` signature recipes is absent.

## Objective (minimize)
The **exact star discrepancy** of the `M`-point set `P`:
```
D*(P) = sup over boxes B = [0,q1) x [0,q2)  of  | (#points of P in B)/M - vol(B) |.
```
The checker computes `D*` exactly over the finite grid of candidate box corners
induced by the point coordinates. Smaller is better.

## Scoring
Let `F` be the star discrepancy of your set and `B` the discrepancy of the checker's
internal baseline (the anchors plus a diagonal fill of the remaining points). The
score is
```
Ratio = min(1000, 100 * B / F) / 1000
```
so reproducing the diagonal baseline scores ~0.1 and a set with one-tenth the
discrepancy caps at 1.0. There is no easy optimum: forced anchors, integer point
count, and the exact discrepancy metric leave a large open design space (lattices,
Hammersley/Sobol sets, golden-ratio shifts, local search).

## Constraints
- `2 <= K <= M`, `8 <= M <= 32`.
- Coordinates are read as floating point; geometry uses a `1e-6` tolerance.

## Example
For `M = 8`, `K = 2`, placing the 6 free points on the diagonal reproduces the
baseline and scores `Ratio: 0.100000`. Replacing them with a Hammersley (base-2)
set spreads the trials over the square, lowers `D*`, and raises the ratio well above
`0.1`.
