# Polar Research Base: Dome Field Around the Antenna

## Problem
A polar research station occupies a square, cleared ice pad modeled as the unit square
`[0,1] x [0,1]`. A central communications mast at the pad center `(0.5, 0.5)` has a
circular exclusion zone of radius `r0` that no structure may enter. You must site `N`
circular living/lab **domes** on the pad. Each dome is a circle; place it to maximize
the total dome radius (a proxy for total usable footprint) while respecting every
safety constraint.

## Input (stdin)
A single line:
```
N r0
```
- `N` — number of domes to place (25 ≤ N ≤ 70).
- `r0` — radius of the central antenna keep-out disk centered at `(0.5, 0.5)` (0.10 ≤ r0 ≤ 0.15).

## Output (stdout)
Exactly `N` lines, one per dome:
```
x y r
```
the center `(x, y)` and radius `r > 0` of each dome (floating point).

## Feasibility (all checked with tolerance 1e-6)
1. **Containment:** each dome lies inside the pad: `x-r ≥ 0`, `x+r ≤ 1`, `y-r ≥ 0`, `y+r ≤ 1`.
2. **Keep-out:** each dome is disjoint from the antenna disk: `dist((x,y),(0.5,0.5)) ≥ r0 + r`.
3. **Non-overlap:** every pair of domes is disjoint: `dist(center_i, center_j) ≥ r_i + r_j`.

Any violation makes the whole submission infeasible (score 0).

## Objective
Maximize `F = sum of all dome radii`.

## Scoring
The checker builds an internal trivial baseline `B` (small equal domes on a coarse
grid, skipping antenna-adjacent cells) and reports
```
Ratio = min(1000, 100 * F / B) / 1000
```
So reproducing the baseline scores ~0.1, and a 10x-better packing caps at 1.0.
The score is fully deterministic.

## Constraints
- Deterministic, tolerance-1e-6 geometry; no randomness in scoring.
- `N` up to 70; an O(N^2) overlap check is used.

## Example
For `N=25, r0=0.11`, a coarse grid of 25 small equal domes (radius `0.3/G`, `G=7`)
placed at cell centers, skipping any cell whose dome would touch the antenna disk,
is feasible and reproduces the baseline (`Ratio ≈ 0.1`). Packing the same 25 domes on
a hexagonal lattice with much larger equal radii raises `F` substantially and scores
well above 0.1.
