# Orbital Debris Cleanup: Sweeper Drone Dispersion

## Problem
A flat orbital debris sector is modelled as the square container `[0, S] x [0, S]`.
You command a fleet of up to `N` autonomous **sweeper drones**. When a drone is parked at
position `(x, y)` it clears all debris inside a circular sweep of radius `r` centred on it.

To avoid drone-to-drone collisions and wasted, redundant coverage, the sweep disks must be
**pairwise non-overlapping** and each must lie **entirely inside** the sector. Fuel and
deorbit-burn budget scale with a drone's sweep radius, and total cleared reach is what mission
control rewards, so you want to make the **total swept radius as large as possible**.

Output a placement of at most `N` sweeper disks that maximizes the sum of their radii.

## Input (stdin)
One line with an integer `N` and a float `S`:
```
N S
```
`N` = number of available drones, `S` = side length of the square sector (`S = 1.0`).

## Output (stdout)
First line: an integer `M` with `0 <= M <= N`, the number of drones you deploy.
Then `M` lines, each `x y r` (floats): a sweep disk centred at `(x, y)` with radius `r >= 0`.

## Feasibility
All checks use tolerance `tol = 1e-6`:
- `0 <= M <= N`, every `r >= -tol`, all numbers finite.
- Containment: `x - r >= -tol`, `x + r <= S + tol`, `y - r >= -tol`, `y + r <= S + tol`.
- Non-overlap: for every pair `i != j`, `dist(center_i, center_j) >= r_i + r_j - tol`.

Any violation scores `Ratio: 0.0`.

## Objective (maximize)
`F = sum of r_i` over the deployed disks.

## Scoring
Let `B` be the checker's internal trivial baseline: `N` equal disks laid in a single row across
the sector, i.e. `B = S / 2`. With `F` your feasible total radius,
```
sc    = min(1000, 100 * F / max(1e-9, B))
Ratio = sc / 1000
```
so the row baseline scores about `0.1` and a placement ten times better caps at `1.0`.

## Constraints
`3 <= N <= 40`, `S = 1.0`. Runs in well under the time limit for these sizes.

## Example
For `N = 4`, `S = 1.0`, four disks of radius `0.25` at the cell centres of a 2x2 grid
(`(0.25,0.25),(0.75,0.25),(0.25,0.75),(0.75,0.75)`) are feasible with `F = 1.0`,
giving `Ratio = min(1000, 100*1.0/0.5)/1000 = 0.2`. The row baseline (four disks of radius
`0.125`) gives `F = 0.5`, `Ratio = 0.1`.
