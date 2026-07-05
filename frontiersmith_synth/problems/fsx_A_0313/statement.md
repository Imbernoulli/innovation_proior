# Solar-Farm Inverter Clearance Packing Around Fixed Keep-Outs

## Problem
A solar farm plot is modelled as the axis-aligned rectangle `[0, W] x [0, H]`.
The plot already contains `M` fixed pieces of infrastructure (transformer pads,
existing panel clusters); seen from above, obstacle `j` is a circle of radius
`oR_j` centred at `(ox_j, oy_j)`. These are **fixed keep-out zones**.

You must site up to `N` new string **inverters**. Each inverter needs a circular
maintenance and thermal **clearance zone**; inverter `i` is a circle of radius
`r_i` centred at `(x_i, y_i)`.

A clearance zone may not extend past the plot fence, may not overlap another
inverter's clearance zone, and may not intrude on any fixed keep-out. Larger
clearance means a bigger inverter with more throughput, and the planner is
rewarded for total installed capacity, so you want to make the **sum of the
clearance radii as large as possible**.

Output a placement of at most `N` inverter clearance circles that maximizes the
sum of their radii.

## Input (stdin)
```
N W H M
ox_1 oy_1 oR_1
...
ox_M oy_M oR_M
```
`N` = number of inverters available, `W` `H` = plot dimensions, `M` = number of
fixed keep-out obstacles, followed by `M` obstacle circles. `M` may be `0`.

## Output (stdout)
First line: an integer `K` with `0 <= K <= N`, the number of inverters you site.
Then `K` lines, each `x y r` (floats): a clearance circle of radius `r >= 0`
centred at `(x, y)`.

## Feasibility
All checks use tolerance `tol = 1e-6`:
- `0 <= K <= N`, every `r >= -tol`, all numbers finite (no `nan`/`inf`).
- Containment in the plot: `x - r >= -tol`, `x + r <= W + tol`,
  `y - r >= -tol`, `y + r <= H + tol`.
- Clear of every keep-out `j`: `dist((x,y),(ox_j,oy_j)) >= r + oR_j - tol`.
- Non-overlap: for every pair `i != j`, `dist(center_i, center_j) >= r_i + r_j - tol`.

Any violation scores `Ratio: 0.0`.

## Objective (maximize)
`F = sum of r_i` over the sited inverters.

## Scoring
Let `B` be the checker's internal trivial baseline: a coarse row-major grid of
`ceil(sqrt(N)) x ceil(N/ceil(sqrt(N)))` cells, one small circle per cell of
radius `0.30 * min(cellW, cellH)`, each shrunk to respect the fence and the fixed
keep-outs; `B` is the sum of those radii. With `F` your feasible total radius,
```
sc    = min(1000, 100 * F / max(1e-9, B))
Ratio = sc / 1000
```
so the trivial grid baseline scores about `0.1`, and a placement ten times better
caps at `1.0`.

## Constraints
`8 <= N <= 36`, `8.0 <= W <= 13.4`, `6.0 <= H <= 9.6`, `0 <= M <= 9`. Obstacle
radii are in `[0.20, 0.55]`. All instances run in well under the time limit.

## Example
Take `N = 4`, `W = 4.0`, `H = 4.0`, `M = 0`. The trivial grid uses a `2 x 2`
layout with `cellW = cellH = 2.0`, so each baseline circle has radius
`0.30 * 2.0 = 0.6` and `B = 4 * 0.6 = 2.4`, `Ratio = 0.1`. Growing each of the
four circles to touch its two fences and its neighbours yields radius `1.0` each
(centres at `(1,1),(3,1),(1,3),(3,3)`), giving `F = 4.0` and
`Ratio = min(1000, 100 * 4.0 / 2.4) / 1000 = 0.1667`. (This 4-circle case is
illustrative; the graded instances are larger and include fixed keep-outs.)
