# Arena Control Zones: Circular-Stage Capture-Point Layout

## Problem
The finals of an e-sports tournament are staged on a **circular arena** modelled as the
disk of radius `R` centred at the origin: `{ (x, y) : x^2 + y^2 <= R^2 }`.

The game director must place up to `N` circular **control zones** (capture points) on the
stage. A zone is a disk `(x, y, r)`. To keep fights readable for the crowd and the cameras,
the zones must be **pairwise non-overlapping**, and every zone must lie **entirely inside**
the arena. The competitive value of a zone scales with its radius (bigger zone = more
ground to contest), and the broadcast rewards total contestable ground, so you want to make
the **sum of the zone radii as large as possible**.

Output a layout of at most `N` zones maximizing the total radius.

## Input (stdin)
One line with an integer `N` and a float `R`:
```
N R
```
`N` = number of available control zones, `R` = arena radius (`R = 1.0`).

## Output (stdout)
First line: an integer `M` with `0 <= M <= N`, the number of zones you place.
Then `M` lines, each `x y r` (floats): a zone centred at `(x, y)` with radius `r >= 0`.

## Feasibility
All checks use tolerance `tol = 1e-6`:
- `0 <= M <= N`, every `r >= -tol`, all numbers finite.
- Containment in the arena: `sqrt(x^2 + y^2) + r <= R + tol` for every zone.
- Non-overlap: for every pair `i != j`, `dist(center_i, center_j) >= r_i + r_j - tol`.

Any violation scores `Ratio: 0.0`.

## Objective (maximize)
`F = sum of r_i` over the placed zones.

## Scoring
Let `B` be the checker's internal trivial baseline: `N` equal zones spread along a diameter,
each of radius `R/N`, so `B = N * (R/N) = R`. With `F` your feasible total radius,
```
sc    = min(1000, 100 * F / max(1e-9, B))
Ratio = sc / 1000
```
so the diameter baseline scores about `0.1` and a layout ten times better caps at `1.0`.

## Constraints
`3 <= N <= 40`, `R = 1.0`. Runs well under the time limit for these sizes.

## Example
For `N = 2`, `R = 1.0`, two zones of radius `0.5` at `(-0.5, 0)` and `(0.5, 0)` are feasible
(each satisfies `0.5 + 0.5 = 1.0 <= R`, and they touch without overlapping) with `F = 1.0`.
Since `B = R = 1.0`, that gives `Ratio = min(1000, 100 * 1.0 / 1.0) / 1000 = 0.1`. Packing
several smaller zones densely into the disk raises `F` well above `R`, lifting the ratio.
