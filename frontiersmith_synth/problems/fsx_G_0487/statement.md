# Corner-Free Reservations: Packing an N x N Slot Grid

## Problem
A conference logistics engine schedules sessions on a grid of **N time slots** (rows,
`0 <= r < N`) by **N tracks** (columns, `0 <= c < N`). Reserving a cell `(r, c)` books track
`c` during slot `r`. Some cells are **blocked** (rooms under maintenance / already committed)
and may not be reserved.

The room-audio routing hardware fails whenever three reserved cells form an axis-aligned
right-isosceles **conflict corner**:

```
(r, c),  (r + d, c),  (r, c + d)      for some integer d >= 1
```

that is, two reservations that share a track separated by `d` slots, together with a third
reservation in the same slot as the first and `d` tracks away. A schedule is **valid** iff it
contains no conflict corner and uses no blocked cell.

Reserve as many cells as possible.

## Input (stdin)
```
N K
r_1 c_1
...
r_K c_K
```
`N` is the grid side. `K` is the number of blocked cells; the next `K` lines list their
coordinates. All coordinates satisfy `0 <= r, c < N`.

## Output (stdout)
```
M
r_1 c_1
...
r_M c_M
```
`M` is the number of cells you reserve, followed by their coordinates (integers). Order does
not matter.

## Feasibility
Your output is rejected (score 0) unless every listed cell is:
- integer-valued and in range `0 <= r, c < N`,
- not a blocked cell,
- distinct (no duplicates),

and the whole set is **corner-free**: there is no `(r, c)`, `(r+d, c)`, `(r, c+d)` with
`d >= 1` all reserved.

## Objective
Maximize `M`, the number of reserved cells (**maximization**).

## Scoring
Let `F` be your reserved count and `B` the checker's baseline: the number of free cells in the
fullest single time slot (a single row is always corner-free). The score is
```
Ratio = min(1000, 100 * F / B) / 1000
```
so matching the single-slot baseline scores `0.1`, and reaching ten times the baseline caps at
`1.0`. Any infeasibility scores `0.0`. The maximum corner-free density is not known in closed
form, so there is genuine headroom above every simple construction.

## Constraints
- `10 <= N <= 28` across the difficulty ladder.
- Deterministic scoring; no ties to time or randomness.

## Example
Take `N = 3` with no blocked cells. The reservation set
```
2
0 0
0 1
```
lies in a single slot, so it is trivially corner-free (score `= 2 / B`). Adding `(1, 0)` would
still be fine, but then also adding `(0, 1)` and `(1, 0)` with `(0,0)` present is safe only if
no third leg closes: reserving `(0,0), (1,0), (0,1)` forms a conflict corner with `d = 1` and
is invalid. A valid 4-cell schedule for `N = 3` is `(0,0),(0,1),(0,2),(1,2)` — check that no
`(r,c),(r+d,c),(r,c+d)` triple is present.
