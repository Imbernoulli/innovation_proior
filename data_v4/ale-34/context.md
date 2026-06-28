# Maze Carving to a Target Difficulty

## Research question

You are given an `H x W` grid of cells. Each cell is either a **wall** (`#`) or an **open corridor**
(`.`). Two distinguished open cells are the **start** `S = (sr, sc)` and the **end** `T = (tr, tc)`.
You are also given a **carving budget** `B`. You must choose **exactly `B` distinct wall cells** and
carve them open. After carving, the score is the **shortest-path length** (number of 4-adjacent steps)
from `S` to `T` over the open cells. **Your objective is to MAXIMIZE that shortest path.**

The tension is that you control which `B` walls become corridor, but you are scored by the *shortest*
route a walker would take afterwards. Spend the budget naively and you dig a short, straight tunnel;
the score is then just the Manhattan distance. To make the maze hard you must lay the open cells so the
only route from `S` to `T` is forced to wind — every carved cell should lengthen the geodesic, and no
carved cell should open a shortcut. This is a constructive optimization over carve-sets with no known
closed form: a budgeted "longest forced detour" problem on a grid.

## Input / output contract

- **Input (stdin):**
  - Line 1: three integers `H W B`.
  - Line 2: four integers `sr sc tr tc` — the start cell `(sr,sc)` and end cell `(tr,tc)`
    (0-indexed, row then column). They are distinct and both open.
  - Next `H` lines: the grid, each a string of exactly `W` characters from `{'#', '.'}`
    (`'#'` = wall, `'.'` = open).
- **Output (stdout):** exactly `B` lines, each `r c` — the wall cell at row `r`, column `c` to carve
  open. The `B` cells must be pairwise distinct, in bounds, and currently walls.
- **Constraints / guarantees:** `20 <= H, W <= 30`. `S` and `T` are open and distinct.
  `1 <= B <= (number of wall cells)`, so carving exactly `B` distinct walls is always possible. The
  budget is chosen large enough (at least the Manhattan distance plus a margin) that a feasible
  carving connecting `S` and `T` always exists.
- **Time limit:** ~2 seconds per instance. **Memory:** 256 MB.

## Background

This is a budgeted variant of the "longest path / maximum-detour" family. The clean fact that drives a
strong heuristic is: if the carved open cells that connect `S` to `T` form an **induced path** — a
sequence of open cells where consecutive cells are 4-adjacent and *non*-consecutive cells are never
adjacent — then the BFS geodesic from `S` to `T` equals the *full* length of that path. There is no
chord to shortcut across, so the shortest path has to traverse every cell on the snake. Maximizing the
score therefore reduces to: grow the longest induced `S`–`T` path you can afford with `B` carved walls.

Two reference approaches frame the problem:

- **Straight-corridor carve (trivial baseline).** Compute the minimum number of walls that must be
  carved to connect `S` and `T` (a 0/1 BFS where stepping onto a wall costs 1), carve that path, then
  spend the leftover budget on arbitrary remaining walls. This connects `S` and `T` but the geodesic is
  essentially the Manhattan distance, and the leftover carves tend to open shortcuts. It is the
  reference the solver must beat.
- **Induced-path construction + simulated annealing (the strong method).** Grow a long induced path
  greedily (a randomized self-avoiding walk that wanders to spend budget, then homes on `T`), keep the
  best over many restarts, then refine with simulated annealing whose moves are *targeted at the
  current geodesic*: remove a carved cell that lies on a shortest path (forcing a detour) and add an
  adjacent wall to keep the carve-count at exactly `B`.

## Evaluation settings

Instances are produced by a fixed **generator** parameterized by an integer seed. For a seed it draws
`H, W in [20,30]`, places `S` near the top-left and `T` near the bottom-right, makes each non-`S`/`T`
cell open with a small probability (`~5%`–`20%`, the rest walls), and sets the budget `B` to a fraction
of the grid area (clamped so `B <= #walls` and `B >=` Manhattan distance `+ 2`). The same seeds are used
for every solver so scores are directly comparable.

A deterministic **scorer** reads the instance and the solver's output and computes the score:

1. Read the listed cells. If their count is not exactly `B`, the score is **0**.
2. For each listed cell: if it is out of bounds, is not currently a wall, or duplicates an earlier cell,
   the score is **0** (the feasibility floor).
3. Carve all `B` cells open. If `S` or `T` is somehow not open, the score is **0**.
4. Run BFS from `S` over the open cells. If `T` is **unreachable**, the score is **0**. Otherwise the
   score is the BFS shortest-path length `S -> T` (an integer `>= 1`).

So **any** infeasibility — wrong count, an illegal carve, or a disconnected result — floors the score to
`0`. A larger reported number is strictly better. We report the mean score over a fixed seed set and
compare against the straight-corridor baseline on the same seeds.

## Code framework

A single self-contained C++17 program that reads the instance from stdin and writes a feasible carve
list to stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int H, W, B, sr, sc, tr, tc;
    if (!(cin >> H >> W >> B)) return 0;
    cin >> sr >> sc >> tr >> tc;
    vector<string> grid(H);
    for (int r = 0; r < H; r++) cin >> grid[r];

    // grid[r][c] == '#' is a wall (carvable); '.' is open. S=(sr,sc), T=(tr,tc) are open.

    // TODO: choose exactly B distinct wall cells to carve so that, over the resulting
    // open cells, the BFS shortest path from S to T is as LONG as possible while S and
    // T stay connected (always keep a feasible, connected carve set on hand).

    vector<pair<int,int>> carved; // the B cells to open

    for (auto& cell : carved) cout << cell.first << ' ' << cell.second << '\n';
    return 0;
}
```
