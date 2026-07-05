# Vaccine Cold-Chain Vial Packing in a Circular Carrier

## Problem
A round, insulated vaccine transport carrier is modelled as the disk of radius `R`
centred at the origin. You must load up to `N` cylindrical **vaccine vials**; seen from
above, vial `i` is a circle of radius `r` centred at `(x, y)`.

Vials may not crush one another and may not poke through the carrier wall, so the vial
circles must be **pairwise non-overlapping** and each must lie **entirely inside** the
carrier disk. The deliverable dose volume of a vial grows with its radius, and the
cold-chain planner is rewarded for total deliverable reach, so you want to make the
**sum of the vial radii as large as possible**.

Output a placement of at most `N` vials that maximizes the sum of their radii.

## Input (stdin)
One line with an integer `N` and a float `R`:
```
N R
```
`N` = number of vials available, `R` = radius of the circular carrier (`R = 1.0`).

## Output (stdout)
First line: an integer `M` with `0 <= M <= N`, the number of vials you load.
Then `M` lines, each `x y r` (floats): a vial circle centred at `(x, y)` with radius `r >= 0`.

## Feasibility
All checks use tolerance `tol = 1e-6`:
- `0 <= M <= N`, every `r >= -tol`, all numbers finite.
- Containment inside the carrier disk of radius `R`: `sqrt(x^2 + y^2) + r <= R + tol`.
- Non-overlap: for every pair `i != j`, `dist(center_i, center_j) >= r_i + r_j - tol`.

Any violation scores `Ratio: 0.0`.

## Objective (maximize)
`F = sum of r_i` over the loaded vials.

## Scoring
Let `B` be the checker's internal trivial baseline: `N` equal vials of radius `R/N` laid
tangent along a diameter (equivalently, a single vial of radius `R`), giving `B = R`.
With `F` your feasible total radius,
```
sc    = min(1000, 100 * F / max(1e-9, B))
Ratio = sc / 1000
```
so the baseline scores about `0.1` and a placement ten times better caps at `1.0`.

## Constraints
`3 <= N <= 40`, `R = 1.0`. Runs in well under the time limit for these sizes.

## Example
For `N = 3`, `R = 1.0`, three mutually tangent equal vials on a ring of radius
`d = R / (1 + sin 60deg) = 0.5359`, each of radius `r = 0.4641`, are feasible with
`F = 3 * 0.4641 = 1.3923`, giving `Ratio = min(1000, 100 * 1.3923 / 1) / 1000 = 0.1392`.
The single-vial baseline (one circle of radius `1.0`) gives `F = 1.0`, `Ratio = 0.1`.
