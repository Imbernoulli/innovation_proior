# Cold-Chain Depot Dispersion: Spreading Refrigerated Hubs

## Problem
A national vaccine program must site `n` refrigerated cold-chain depots across a
square service region (normalised to the unit square `[0,1]^2`). To keep the network
resilient — a power failure or road closure at one depot should never leave a whole
cluster of depots dark at once — the depots must be **spread out**: the planners care
about the *closest* pair of depots. You want to place the depots so that the
**minimum pairwise distance** between any two depots is as large as possible.

Complicating things, some parts of the region are **warm zones** (open disks) where
ambient conditions make refrigeration infeasible — no depot may be sited strictly
inside a warm zone.

This is a **max-min dispersion** (spreading-points) placement, a classic extremal
point-configuration problem: it is NP-hard in general, has no easy optimum, and admits
many distinct strategies (grid layouts, farthest-point greedy, local search, annealing).

## Input (stdin)
```
n
c
<c lines, each: cx cy r>
```
- `n` — number of depots to place.
- `c` — number of warm zones.
- each warm zone is an open disk of centre `(cx, cy)` and radius `r`.

## Output (stdout)
```
<n lines, each: x y>
```
Print the `n` depot coordinates, one `x y` pair per line (real numbers).

## Feasibility
An output is valid iff **all** hold (tolerance `1e-6`):
- exactly `n` coordinate pairs are printed;
- every depot lies in the service region: `0 <= x <= 1`, `0 <= y <= 1`;
- no depot lies strictly inside any warm-zone disk (`dist((x,y),(cx,cy)) >= r`);
- all `n` depots are distinct.

Any violation scores `Ratio: 0.0`.

## Objective
Maximize
```
F = min over all pairs {i,j} of  dist(depot_i, depot_j)
```
where `dist` is Euclidean distance.

## Scoring
The checker builds its own baseline `B`: the single-row layout that places the `n`
depots on the horizontal midline,
```
x_i = (i+0.5)/n,  y_i = 0.5   (i = 0..n-1)
```
whose minimum pairwise distance is exactly `B = 1/n`. The generator guarantees this
row is always feasible (warm zones never touch the midline). With maximization
normalization:
```
sc    = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Reproducing the row scores `Ratio = 0.1`; a layout whose minimum pairwise distance is
`10x` the row baseline caps at `1.0`.

## Constraints
- `13 <= n <= 22`.
- `2 <= c <= 6`; warm-zone radii lie in `[0.04, 0.10]`; every warm zone keeps its
  centre at least `r + 0.03` from the midline `y = 0.5` (so the baseline row is always
  feasible).
- Time limit 5s, memory 512m.

## Example
For `n = 16` the row baseline has minimum pairwise distance `B = 1/16 = 0.0625` and
scores `Ratio = 0.1`. A 4x4 grid layout spreads the depots so the closest pair is,
say, `0.30` apart, giving `sc = 100 * 0.30 / 0.0625 = 480`, `Ratio = 0.480`.
