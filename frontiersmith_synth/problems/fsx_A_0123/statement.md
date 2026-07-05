# Data-Center Cooling: Cold-Air Diffuser Placement

## Problem
A data hall is modelled as the rectangular raised floor `[0, W] x [0, H]`. You install up to
`N` circular **cold-air diffusers**. A diffuser centred at `(x, y)` with throw radius `r` cools
the disk of radius `r` around it.

The floor already holds `K` **server racks**, each an axis-aligned rectangular **keep-out zone**
(no diffuser may be cut into a rack, and its cold-air disk must not reach inside one). To avoid
turbulent interference the diffuser disks must be **pairwise non-overlapping**, and every disk
must lie **entirely inside the room**. Cooling reach (and the CRAH budget it justifies) grows
with a diffuser's throw radius, so you maximize the **total throw radius** of all diffusers.

Output a placement of at most `N` diffuser disks that maximizes the sum of their radii.

## Input (stdin)
```
N W H K
x0 y0 x1 y1        (rack 1)
...                (K rack lines total)
```
`N` = number of available diffusers, `W`,`H` = room dimensions, `K` = number of server racks.
Each rack line gives its lower-left `(x0,y0)` and upper-right `(x1,y1)` corner with
`0 <= x0 < x1 <= W`, `0 <= y0 < y1 <= H`. Racks are pairwise disjoint.

## Output (stdout)
First line: an integer `M` with `0 <= M <= N`, the number of diffusers you install.
Then `M` lines, each `x y r` (floats): a diffuser centred at `(x, y)` with radius `r >= 0`.

## Feasibility
All checks use tolerance `tol = 1e-6`:
- `0 <= M <= N`, every `r >= -tol`, all numbers finite.
- Containment: `x - r >= -tol`, `x + r <= W + tol`, `y - r >= -tol`, `y + r <= H + tol`.
- Rack clearance: for each rack, the distance from `(x, y)` to the rack rectangle is `>= r - tol`
  (a centre inside a rack has distance `0`, so any positive radius there is infeasible).
- Non-overlap: for every pair `i != j`, `dist(center_i, center_j) >= r_i + r_j - tol`.

Any violation scores `Ratio: 0.0`.

## Objective (maximize)
`F = sum of r_i` over the installed diffusers.

## Scoring
Let `B` be the checker's internal baseline: an obstacle-aware single centre row of `N` touching
equal diffusers (radius `min(W/(2N), H/2)`), dropping any disk that would hit a rack or a wall.
With `F` your feasible total radius,
```
sc    = min(1000, 100 * F / max(1e-9, B))
Ratio = sc / 1000
```
so the grid baseline scores about `0.1` and a placement ten times better caps at `1.0`.

## Constraints
`10 <= N <= 40`, `1.0 <= W <= 1.5`, `0.6 <= H <= 1.0`, `1 <= K <= 5`. Runs well under the
time limit for these sizes.

## Example
For `N = 12`, `W = 1.0`, `H = 0.75`, `K = 0` (no racks), a `4 x 3` grid of touching disks of
radius `0.125` gives `F = 12 * 0.125 = 1.5`. Growing the four corner disks toward the walls, or
inserting a few large disks into the gaps, raises `F` above the baseline `B` and lifts the ratio
past `0.1`.
