# Drone Delivery Swarm: Safety-Bubble Packing in Restricted Airspace

## Problem
A drone-delivery operator manages a swarm hovering inside a **circular delivery airspace** — a
disk of radius `R` centred at `(cx, cy)`. The airspace contains `K` fixed **no-fly zones**
(restricted-airspace circles) that no drone may enter.

Each drone reserves a circular **safety bubble** around itself. You must choose positions and
bubble radii for up to `N` drones so that the bubbles

- lie entirely inside the airspace disk,
- do not intersect any no-fly zone, and
- do not overlap one another,

while making the **total safety margin** — the sum of all bubble radii — as large as possible.
A larger total radius means more collision-avoidance slack for the swarm.

## Input (stdin)
```
N cx cy R K
zx_1 zy_1 zr_1
...
zx_K zy_K zr_K
```
- `N`  — maximum number of drone bubbles you may place.
- `cx cy R` — centre and radius of the circular airspace disk.
- `K`  — number of no-fly zones; each line gives a zone circle `(zx, zy)` radius `zr`.

## Output (stdout)
```
M
x_1 y_1 r_1
...
x_M y_M r_M
```
Print an integer `M` with `0 <= M <= N`, then `M` bubbles as centre `(x, y)` and radius `r`.
You may place fewer than `N` bubbles. Radii must be `>= 0`.

## Feasibility (tolerance 1e-6)
A submission is feasible iff, for every bubble `(x, y, r)`:
- containment: `dist((x,y),(cx,cy)) + r <= R`,
- no-fly clearance: for every zone `j`, `dist((x,y),(zx_j,zy_j)) >= r + zr_j`,
- non-overlap: for every other bubble `k`, `dist((x,y),(x_k,y_k)) >= r + r_k`.

Any violation scores `Ratio: 0.0`.

## Objective
Maximize `F = sum(r_i)` over all placed bubbles.

## Scoring
The checker builds an internal baseline `B` (an airspace-aware equal-radius grid of touching
bubbles). Your score is
```
Ratio = min(1.0, 0.1 * F / B)
```
so reproducing the baseline gives `~0.1` and a 10x-better total caps at `1.0`.

## Constraints
- `20 <= N <= 56`, `1 <= K <= 5`, `1.0 <= R <= 2.1`.
- All coordinates/radii are real numbers; geometry is checked with a fixed `1e-6` tolerance.

## Example (worked score)
Suppose `N=20` and the equal-radius grid baseline yields `B = 3.0`. If your construction places
bubbles totalling `F = 6.0` in radius, then `Ratio = min(1.0, 0.1 * 6.0 / 3.0) = 0.2`. Doubling
the total safety margin over the naive grid doubles the score.
