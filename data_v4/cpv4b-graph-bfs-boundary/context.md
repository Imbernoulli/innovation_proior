# Frost band in a greenhouse (multi-source grid BFS, inclusive distance window)

## Research question

A greenhouse floor is an `R x C` grid. Each cell is one of:

- `.` an open growing cell,
- `#` a solid partition wall,
- `*` an open cell that also holds a **frost vent**.

On a cold night frost spreads outward from every vent simultaneously. It moves between
edge-adjacent open cells (up/down/left/right), one cell per minute; it cannot pass through a wall
`#`, and walls themselves are never frosted. The **frost time** of an open cell is the number of
minutes until frost first reaches it — that is, its shortest-path distance (in grid steps) to the
nearest vent. A vent cell has frost time `0`.

A plant is planted on every open cell. A plant is **damaged** exactly when its frost time `d`
satisfies `L <= d <= U` for a given **inclusive** window `[L, U]`: plants reached too early
(`d < L`, still near a vent where keepers pre-warm the soil) and plants reached too late
(`d > U`, the frost has weakened, or it never arrives) survive. Count the damaged open cells.

The whole problem turns on the boundary of that window: the window is inclusive on *both* ends, the
distance is measured in steps with the vent at `0`, and an open cell the frost never reaches must
not be counted. A single `<` where `<=` belongs, or counting the source as if it were one step out,
silently changes the answer on a small grid — so this is a place where one off-by-one decides
correctness.

## Input / output contract

- Input (stdin):
  - First line: four integers `R C L U`
    (`1 <= R, C <= 1000`, `0 <= L <= U <= R*C`).
  - Next `R` lines: each a string of exactly `C` characters drawn from `{'.', '#', '*'}`.
- Output (stdout): a single line with the number of open cells whose frost time `d` lies in the
  inclusive window `[L, U]`. Walls are never counted; open cells unreachable from any vent are
  never counted. If there are no vents at all, every open cell is unreachable and the answer is `0`.
- Time limit: 1 second. Memory: 256 MB.

Example. Grid (`R=4, C=5`), window `[1, 2]`:

```
.....
.*.#.
.....
..*..
```

The frost times (with `X` marking the wall) are

```
2 1 2 3 4
1 0 1 X 4
2 1 1 2 3
2 1 0 1 2
```

so the answer is `13`: every open cell with `d in {1, 2}`. (For the same grid, window `[0, 0]`
gives `2` — only the two vents; window `[0, 2]` gives `15`.)

## Background

This is a multi-source breadth-first search on a grid graph, followed by a count restricted to a
distance band. Two design points are worth naming before committing:

- **Multi-source BFS in one pass vs. per-source BFS.** Seeding the queue with *all* vents at
  distance `0` and running a single BFS yields, for every cell, the distance to its nearest vent in
  `O(R*C)` total — the queue stays distance-sorted because every edge has weight `1`. The
  alternative, one BFS per vent followed by an element-wise minimum, is `O(R*C)` per vent and blows
  up when many cells are vents. The single-pass version is the one that scales to `R*C = 10^6`.
- **The band test.** With distances in hand the count is a filter `L <= d <= U`. The subtlety is
  entirely in the predicate and in what "reached" means: the inclusive endpoints, the vent sitting
  at `d = 0`, and the unreachable cells (distance never assigned) that must be excluded even when
  `L = 0`.

## Evaluation settings

Judged on hidden tests covering: grids with no vents (answer `0`); a single vent; many vents;
windows that touch the boundaries (`L = 0`, `U = 0`, `L = U`, `U >= R*C`); grids that are fully
walled-off pockets so some open cells are unreachable; thin `1 x C` and `R x 1` grids; and a full
`1000 x 1000` grid (so an `O(R*C)` BFS is required and the count can exceed a 32-bit-friendly range
only mildly, but a 64-bit counter is used for safety).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int R, C, L, U;
    if (!(cin >> R >> C >> L >> U)) return 0;
    vector<string> g(R);
    for (auto &row : g) cin >> row;

    // TODO: multi-source BFS from every '*' to get each open cell's frost time,
    //       then count open cells whose time d satisfies the inclusive band L <= d <= U.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
