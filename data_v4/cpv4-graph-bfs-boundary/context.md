# Counting cells in a patrol band (multi-source BFS, inclusive distance ring)

## Research question

A city block is given as an `H x W` grid of cells. Each cell is one of:

- `.` an open cell you may walk through,
- `#` a wall you may never enter,
- `S` an open cell that also hosts a **patrol station**.

From any open cell you may step to one of its four orthogonal neighbours, provided that
neighbour is inside the grid and is not a wall. For every open cell define `d(cell)` = the
minimum number of steps needed to reach it from the **nearest** patrol station (a station is at
distance `0` from itself), travelling only through open cells. Some open cells may be unreachable
from every station; those have `d = infinity`.

You are also given two integers `L` and `R` with `0 <= L <= R`. A cell is said to lie in the
**patrol band** if it is open, reachable, and its nearest-station distance satisfies the *inclusive*
condition `L <= d(cell) <= R`. Count how many cells lie in the patrol band and print that count.

The phrase "inclusive band `[L, R]`" is the crux: a station itself (`d = 0`) belongs to the band
exactly when `L = 0`, and the outermost ring `d = R` is part of the band, not one step past it.
This is the kind of distance-thresholded counting that shows up in coverage, flood-fill, and
service-radius problems, where one inclusive-vs-exclusive slip on either end of the range silently
miscounts a whole ring of cells.

## Input / output contract

- Input (stdin):
  - line 1: two integers `H` and `W` (`1 <= H, W <= 1000`),
  - line 2: two integers `L` and `R` (`0 <= L <= R <= H*W`),
  - next `H` lines: each a string of exactly `W` characters from `{'.', '#', 'S'}`.
  - At least one `S` is guaranteed to appear in the grid.
- Output (stdout): a single line with the number of open, reachable cells whose nearest-station
  distance `d` satisfies `L <= d <= R`.
- Time limit: 1 second. Memory: 256 MB.

Worked example 1:

```
5 6
1 2
......
.S..#.
..#...
....S.
#.....
```

The nearest-station distances (with `#` for walls) are

```
2 1 2 3 4 4
1 0 1 2 # 3
2 1 # 2 1 2
3 2 2 1 0 1
# 3 3 2 1 2
```

With `L = 1`, `R = 2` we count every cell whose distance is `1` or `2`. The two stations
(distance `0`) are excluded because `L = 1`; the cells at distance `3` and `4` are excluded
because `R = 2`. The remaining cells number `18`, so the answer is `18`.

Worked example 2 (the `L = 0` corner, where stations count):

```
3 4
0 1
S..#
....
#..S
```

Distances:

```
0 1 2 #
1 2 2 1
# 2 1 0
```

With `L = 0`, `R = 1` we count distances `0` and `1`: both stations plus their immediate open
neighbours, which is `6` cells. The answer is `6`.

## Background

The distance map is a classic **multi-source breadth-first search**: seed a queue with every station
at distance `0` and relax outward; because all edges have unit weight, the first time BFS settles a
cell it has the minimum distance to the nearest seed. That part is standard. The danger is entirely
at the **counting boundary**:

- *Low end.* `d = L` is inside the band. When `L = 0`, stations themselves must be counted; an
  off-by-one that uses `d > L` would drop the entire inner ring.
- *High end.* `d = R` is inside the band. Writing `d < R` instead of `d <= R` silently omits the
  outermost ring — a whole layer of cells — and this is invisible on tiny hand examples where the
  outer ring happens to be empty.
- *Unreachable cells.* Open cells walled off from every station have `d = infinity` and must never be
  counted; a careless `d <= R` over a distance array left at a large sentinel could miscount them if
  the sentinel is finite, or if `R` is large enough to swallow the sentinel.
- *Walls.* Wall cells are never counted regardless of `L`, `R`.

Choosing bounds `R` up to `H*W` makes the high-end inclusive/exclusive choice observable on
non-degenerate inputs, and allowing `L = 0` makes the station-inclusion choice observable too.

## Evaluation settings

Judged on hidden tests covering: `L = 0` (stations counted), `L = R` (a single exact ring),
`R` at the diameter of the grid (outermost ring non-empty, so `d < R` vs `d <= R` differ),
grids with unreachable open pockets behind walls, a `1 x 1` grid, fully open grids, grids that are
almost all walls, and large `1000 x 1000` grids where an `O(H*W)` BFS is required and an `O(n^2)`
all-pairs approach would time out.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int H, W;
    if (!(cin >> H >> W)) return 0;
    int L, R;
    cin >> L >> R;

    vector<string> g(H);
    for (int i = 0; i < H; i++) cin >> g[i];

    // TODO: multi-source BFS from all 'S' cells to get nearest-station distance per open cell,
    //       then count open, reachable cells with L <= d <= R (inclusive on BOTH ends).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
