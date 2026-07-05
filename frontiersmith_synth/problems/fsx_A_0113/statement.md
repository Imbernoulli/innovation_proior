# Power-Grid Substation Siting: Weighted Coverage Packing

## Problem
A square service region, the unit square `[0,1] x [0,1]`, must host `N` electrical
substations. Substation `i` carries a priority load (weight) `w_i > 0`. Each
substation projects a circular **coverage disk** of radius `r_i` centred at its
location `(x_i, y_i)`. Two coverage disks may not overlap (electromagnetic
clearance), and every disk must lie fully inside the region.

Choose the locations and radii to maximise the total **weighted coverage**, so
that high-load substations are given the largest coverage disks.

## Input (stdin)
```
N
w_1 w_2 ... w_N
```
`N` substations (11..20). Each `w_i` is a positive integer (1..100). The weight
distribution is heavily skewed: a few high-load substations among many small ones.

## Output (stdout)
`N` lines, one per substation in input order:
```
x_i y_i r_i
```
`x_i, y_i` is the centre and `r_i >= 0` the coverage radius of substation `i`.

## Feasibility (tolerance 1e-6)
- Containment: `r_i <= x_i <= 1 - r_i` and `r_i <= y_i <= 1 - r_i`.
- Non-negative radii: `r_i >= 0`.
- Non-overlap: for all `i != j`, `dist((x_i,y_i),(x_j,y_j)) >= r_i + r_j - 1e-6`.

Any violation scores `Ratio: 0.0`.

## Objective
Maximise `F = sum_i w_i * r_i`.

## Scoring
The checker builds an internal uniform-grid baseline `B`: a
`k x k` grid with `k = ceil(sqrt(N))`, every disk radius `1/(2k)`, so
`B = (sum_i w_i) / (2k)`. Then
```
sc    = min(1000, 100 * F / max(1e-9, B))
Ratio = sc / 1000
```
The grid layout scores `Ratio ~ 0.1`; a layout ten times better caps at `1.0`.

## Constraints
- `11 <= N <= 20`, `1 <= w_i <= 100`.
- Deterministic scoring only.

## Example
For `N = 2`, `w = [100, 1]`: placing the heavy substation at `(0.25, 0.5)` with
`r = 0.25` and the light one at `(0.75, 0.5)` with `r = 0.25` gives
`F = 100*0.25 + 1*0.25 = 25.25`. The grid baseline (`k = 2`, `r = 0.25`) gives
`B = 101 * 0.25 = 25.25`, so `Ratio = 0.1`. Shrinking the light disk and enlarging
the heavy one (e.g. heavy `r = 0.35` off in a corner, light `r` smaller) raises
`F` above `B` and pushes `Ratio` upward.
