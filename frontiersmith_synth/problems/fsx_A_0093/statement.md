# Rotunda Gallery Tour: Viewing-Zone Layout

## Problem
A circular museum rotunda has an outer wall modelled as the disk of radius `R` centred at the
origin. In the very middle stands a **central sculpture** whose protective cordon is the disk
of radius `r_in` (also centred at the origin). Visitors may only stand in the **annular gallery
floor** between the cordon and the outer wall.

You are laying out up to `N` circular **viewing zones** (soft rugs on the floor). Each viewing
zone is a disk `(x, y, r)`. A layout is valid when every viewing zone lies **entirely on the
gallery floor** — inside the outer wall and outside the central cordon — and the zones are
**pairwise non-overlapping** (a visitor in one zone never bumps a visitor in another).

Larger zones let more visitors gather, so mission control rewards the **total zone radius**.
Output a layout of at most `N` viewing zones that maximizes the sum of their radii.

## Input (stdin)
One line with an integer `N` and two floats `R` and `r_in`:
```
N R r_in
```
`N` = number of available viewing zones, `R` = outer wall radius, `r_in` = central cordon radius
(`0 < r_in < R`).

## Output (stdout)
First line: an integer `M` with `0 <= M <= N`, the number of viewing zones you place.
Then `M` lines, each `x y r` (floats): a viewing zone centred at `(x, y)` with radius `r >= 0`.

## Feasibility
All checks use tolerance `tol = 1e-6`:
- `0 <= M <= N`, every `r >= -tol`, all numbers finite.
- Outer containment: `sqrt(x^2 + y^2) + r <= R + tol`.
- Cordon clearance: `sqrt(x^2 + y^2) - r >= r_in - tol`.
- Non-overlap: for every pair `i != j`, `dist(center_i, center_j) >= r_i + r_j - tol`.

Any violation scores `Ratio: 0.0`.

## Objective (maximize)
`F = sum of r_i` over the placed zones.

## Scoring
Let `B` be the checker's internal trivial baseline: a **single ring** of `N` equal zones evenly
spaced on the mid-annulus circle of radius `rmid = (r_in + R) / 2`. Each ring zone has radius
`r_base = min((R - r_in) / 2, rmid * sin(pi / N))` (the smaller of the radial half-width and the
angular half-gap), so `B = N * r_base`. With `F` your feasible total radius,
```
sc    = min(1000, 100 * F / max(1e-9, B))
Ratio = sc / 1000
```
so the single-ring baseline scores about `0.1`, and a layout ten times better caps at `1.0`.

## Constraints
`10 <= N <= 40`, `R = 1.0`, `0.15 < r_in < 0.4`. Runs well under the time limit for these sizes.

## Example
For `N = 12`, `R = 1.0`, `r_in = 0.2`, the single-ring baseline places 12 zones on the circle of
radius `rmid = 0.6`. Here `rmid * sin(pi/12) = 0.6 * 0.2588 = 0.1553` is smaller than
`(R - r_in)/2 = 0.4`, so `r_base = 0.1553` and `B = 12 * 0.1553 = 1.863`, giving `Ratio = 0.1`.
A layout that also fills a second inner ring can push `F` well above `B` and raise the ratio.
