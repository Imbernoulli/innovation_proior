# Airtanker Isochrone Cut

## Problem
A wildfire starts on a grid of `R` rows by `C` columns. Every cell `(r,c)` has a
terrain **delay** `delay[r][c] >= 1`: once any orthogonal neighbor of `(r,c)` is on
fire at time `t`, cell `(r,c)` ignites at exactly `t + delay[r][c]` (the earliest such
time over all burning neighbors), unless it has been **fireproofed** by then. A set of
cells is on fire at time `0`. Some cells are **lakes**: fire never enters them, and they
are the only places a plane may refill.

You control `K` airtankers. Plane `i` starts at a given cell with a given **tank
capacity** (max drops before refuelling). Every plane shares one **move cost** `move`
(ticks per grid cell of Manhattan travel) and one **refill duration** `Rfill` (ticks,
once a refill slot is acquired). Each lake has a single refill slot: if a plane arrives
while the slot is occupied, it must wait until the plane ahead finishes (queueing).
A drop is instantaneous on arrival and permanently fireproofs that cell -- but a drop
landing on a cell that is *already* on fire at that moment (or later) has no effect: the
load is wasted.

Your program reads the full instance (terrain, origins, lakes, planes) and outputs, for
each plane, a timed sequence of waypoints. The checker replays the fire spread and every
plane's itinerary (with real refill queueing) and scores the total burned area at a fixed
horizon `Tmax`.

## Input (stdin)
```
R C Tmax
move Rfill
S
r c            (S lines: fire-origin cells, on fire at t=0)
L
r c            (L lines: lake cells)
K
r c tank       (K lines: plane start row, col, tank capacity)
R lines of C integers: delay[r][c]  (a very large value is effectively unreachable
                                      within Tmax -- treat it as impassable to fire)
```

## Output (stdout)
```
K
m_1
r c A          (m_1 lines: A is 'D' (drop/fireproof) or 'F' (refill))
m_2
...
```
Print `K` (must match the input), then for each plane in input order its waypoint count
followed by that many `r c A` lines, visited in order.

## Feasibility
An output is valid iff **all** hold:
- the plane count matches the input;
- every waypoint cell is in bounds;
- an `F` waypoint's cell is exactly a lake cell;
- a `D` waypoint's cell is neither a lake nor a fire-origin cell;
- no plane ever attempts a `D` waypoint while its tank is empty (a plane's tank resets to
  full only on completing a refill).
Any violation scores `Ratio: 0.0`.

## Objective
Let `F` be the number of cells with true ignite time `<= Tmax` after replaying your
itinerary (fireproofed-in-time cells never ignite and never propagate fire further).
**Minimize `F`.**

## Scoring
Let `B` be the checker's own baseline: the burned area at `Tmax` if no drops are made at
all (always a valid, achievable reference). With minimization normalization:
```
sc = min(1000.0, 100.0 * B / max(1e-9, F))
Ratio = sc / 1000.0
```
Doing nothing reproduces `B` and scores `Ratio = 0.1`; cutting the burn to a tenth of
baseline caps at `1.0`.

## Constraints
- `8 <= R,C <= 40`, `1 <= K <= 8`, `1 <= L <= 3`.
- `delay[r][c] >= 1` everywhere; `move, Rfill, tank >= 1`.
- Time limit 5s, memory 512MB.

## Example
Suppose doing nothing burns `B = 200` cells by `Tmax`. A player who reactively
fireproofs whichever cell is about to ignite next -- without any look-ahead -- typically
only dents an open field (fire routes around isolated blocked cells) and reaches
`F = 160`, `Ratio = 100*200/160/1000 = 0.125`. A player who instead computes the full
fire-arrival-time map, finds a later, narrower isochrone that the fleet can fully seal
(accounting for travel time and lake-refill queueing) before the fire reaches it, and
fireproofs that whole boundary at once might reach `F = 70`, `Ratio = 100*200/70/1000
= 0.286` -- because a complete cut, made far enough out that its perimeter matches what
the fleet can finish in time, stops far more fire than the same number of scattered,
reactive drops ever could.
