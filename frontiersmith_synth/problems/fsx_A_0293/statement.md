# Reservoir Dam Network: Maximizing Dam Frontage Around a Protected Wetland

## Problem
A flat valley floor is modelled as the rectangle `[0, W] x [0, H]`. You are planning a
network of up to `N` circular **reservoirs**, each impounded by a ring dam. A reservoir
centred at `(x, y)` with radius `r` occupies the disk of radius `r` about that point.

At the centre of the valley lies a **protected wetland** — a circular exclusion zone
centred at `(CX, CY)` with radius `Q` — which no reservoir may flood. To prevent dam
failure and shared-embankment collapse, reservoir disks must be **pairwise
non-overlapping**, must lie **entirely inside** the valley floor, and must **not overlap
the wetland**.

The total length of dam wall you can license scales with the sum of the reservoir radii
(total *dam frontage*), which is what the water authority rewards. Output a placement of at
most `N` reservoir disks that maximizes the sum of their radii.

## Input (stdin)
One line with six numbers:
```
N W H CX CY Q
```
`N` = reservoir budget (integer); `W`,`H` = valley dimensions; `(CX, CY)` = wetland centre;
`Q` = wetland radius.

## Output (stdout)
First line: an integer `M` with `0 <= M <= N`, the number of reservoirs you build.
Then `M` lines, each `x y r` (floats): a reservoir disk centred at `(x, y)` with radius
`r >= 0`.

## Feasibility
All checks use tolerance `tol = 1e-6`:
- `0 <= M <= N`, every `r >= -tol`, all numbers finite (no `nan`/`inf`).
- Containment: `x - r >= -tol`, `x + r <= W + tol`, `y - r >= -tol`, `y + r <= H + tol`.
- Wetland clearance: `dist((x,y),(CX,CY)) >= r + Q - tol`.
- Non-overlap: for every pair `i != j`, `dist(center_i, center_j) >= r_i + r_j - tol`.

Any violation scores `Ratio: 0.0`.

## Objective (maximize)
`F = sum of r_i` over the deployed reservoirs.

## Scoring
Let `B` be the checker's internal baseline: a single bottom row of `N` equal disks that
clears the wetland, with radius `r_b = min(W/(2N), (H/2 - Q)/2)` and `B = N * r_b`. With
`F` your feasible total radius,
```
sc    = min(1000, 100 * F / max(1e-9, B))
Ratio = sc / 1000
```
so the row baseline scores about `0.1` and a placement ten times better caps at `1.0`.

## Constraints
`12 <= N <= 90`, `W = 2.0`, `H = 1.0`, wetland centred at `(1.0, 0.5)` with
`0.15 <= Q <= 0.35`. The instance always satisfies `H/2 - Q >= W/N`, so the row baseline is
feasible. Runs comfortably within the time limit for these sizes.

## Example
Illustrative only. Suppose `N = 4`, `W = 2.0`, `H = 1.0`, wetland at `(1.0, 0.5)`,
`Q = 0.2`. Four reservoirs of radius `0.2` centred near the four corners
(`(0.2,0.2),(1.8,0.2),(0.2,0.8),(1.8,0.8)`) are pairwise disjoint, inside the valley, and
each clears the wetland (nearest-corner distance to centre `~1.13 > 0.2 + 0.2`), giving
`F = 0.8`. The bottom-row baseline places four disks of radius
`r_b = min(2/8, (0.5-0.2)/2) = min(0.25, 0.15) = 0.15`, so `B = 4 * 0.15 = 0.6` and the
baseline `Ratio = min(1000, 100*0.6/0.6)/1000 = 0.1`. The corner placement scores
`Ratio = 100*0.8/0.6/1000 = 0.133`.
