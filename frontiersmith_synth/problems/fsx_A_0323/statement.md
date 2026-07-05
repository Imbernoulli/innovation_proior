# Highway Toll Gantries: Sensor Coverage on the Strip

## Problem
A long stretch of tolled highway is modelled as the thin strip `[0, L] x [0, W]`
(`W = 1.0`, `L` long). You must deploy up to `N` **toll-gantry sensor stations**. A
station parked at `(x, y)` monitors a circular zone of radius `r` centred on it.

Sensor zones may not interfere: they must be **pairwise non-overlapping** and lie
**entirely inside the strip**. The roadway also carries `K` fixed **support pylons**,
each a small disk of radius `rho` at a known position; a sensor zone may **not overlap
any pylon** (a station cannot see through a pylon). Revenue-grade coverage scales with a
station's zone radius, so you want the **total monitored radius as large as possible**.

Output a placement of at most `N` sensor disks maximizing the sum of their radii.

## Input (stdin)
```
N L W K rho
px_1 py_1
px_2 py_2
...
px_K py_K
```
`N` sensor budget, strip length `L`, strip width `W = 1.0`, `K` pylons of radius `rho`,
then `K` lines giving each pylon centre `(px_j, py_j)`. All pylons sit in the two edge
bands near `y = 0` and `y = W`; the central corridor is always clear.

## Output (stdout)
First line: an integer `M` with `0 <= M <= N`, the number of stations deployed.
Then `M` lines, each `x y r` (floats): a sensor zone centred at `(x, y)` with radius
`r >= 0`.

## Feasibility
All checks use tolerance `tol = 1e-6`:
- `0 <= M <= N`, every `r >= -tol`, all numbers finite (no `nan`/`inf`).
- Containment: `x - r >= -tol`, `x + r <= L + tol`, `y - r >= -tol`, `y + r <= W + tol`.
- Non-overlap (sensors): for every pair `i != j`, `dist(c_i, c_j) >= r_i + r_j - tol`.
- Pylon clearance: for every disk `i` and pylon `j`, `dist(c_i, p_j) >= r_i + rho - tol`.

Any violation scores `Ratio: 0.0`.

## Objective (maximize)
`F = sum of r_i` over the deployed disks.

## Scoring
Let `B` be the checker's internal baseline: `N` equal disks in a single centerline row,
`r0 = min(W/4, L/(2N))`, `B = N * r0`. With `F` your feasible total radius,
```
sc    = min(1000, 100 * F / max(1e-9, B))
Ratio = sc / 1000
```
so the single-row baseline scores about `0.1` and a placement ten times better caps at
`1.0`. Using the road width (multiple rows) and the pylon-free gaps beats the baseline,
but boundaries, the disk budget, and the pylons keep the optimum well short of the cap.

## Constraints
`20 <= N <= 120`, `W = 1.0`, `L = 0.15 * N`, `K = max(3, N // 8)`, `rho = 0.06`.
Runs in well under the time limit for these sizes.

## Example
Illustrative shape only. For `N = 2`, `L = 3`, `W = 1`, no pylons in the corridor: two
disks of radius `0.25` at `(0.75, 0.5)` and `(2.25, 0.5)` are feasible with `F = 0.5`.
The single-row baseline uses `r0 = min(0.25, 0.75) = 0.25` and `B = 0.5`, so both give
`Ratio = min(1000, 100*0.5/0.5)/1000 = 0.1`. Widening into two rows so each disk can grow
past `0.25` is what pushes the ratio above the baseline.
