# Shortest path on a state-product grid: minimum walls broken with a budget

## Research question

You are given a grid of `R` rows and `C` columns. Each cell is one of: `.` (a free cell you may
walk on), `#` (a wall), `S` (the unique start cell, also free), or `T` (the unique target cell, also
free). You stand on `S` and want to reach `T`, moving one step at a time to one of the four
orthogonally adjacent cells (up, down, left, right) that lies inside the grid.

Normally you may only step onto free cells. You are additionally given a budget `K`: you may **break
up to `K` walls** over the whole journey. Stepping onto a free cell (`.`, `S`, or `T`) consumes no
budget; stepping onto a `#` cell costs **one** break and is allowed only while you still have budget.

Output the **minimum number of walls you must break** to get from `S` to `T`, or `-1` if `T` is
unreachable even when you spend all `K` breaks. The candidate "you may flip ≤ 1 wall" is the special
case `K = 1`; the general `K` makes the state structure explicit.

## Input / output contract

- Input (stdin):
  - The first line has three integers `R C K` (`1 <= R, C <= 1000`, `0 <= K <= 10`).
  - The next `R` lines each contain a string of exactly `C` characters drawn from `{'.', '#', 'S', 'T'}`.
  - Exactly one `S` and exactly one `T` appear in the grid.
- Output (stdout): a single line with the minimum number of walls broken to reach `T` from `S`, or
  `-1` if it cannot be reached using at most `K` breaks.
- Time limit: 1 second. Memory: 256 MB.

Example: for the grid

```
S#.T
.#..
....
```

with `K = 1`, the answer is `0`: a wall-free path exists (down from `S`, along the bottom row, then up
into `T`), so no wall needs to be broken even though one break is available.

## Background

The constraint that the *same physical cell* can be entered "with budget still in hand" or "with
budget already spent" makes this more than a plain grid BFS. Two ideas are on the table before
committing to one:

- **One BFS on the raw grid, treating walls as free once you decide to break them.** The trouble is
  that "how many breaks remain" is part of the state of the search, not a property of the cell: the
  same cell reached with `0` breaks used versus `1` break used can lead to different futures. A flat
  BFS that stores one distance per cell cannot represent both.
- **Search over an enlarged state.** Make the search node be `(cell, breaks_used)`. Every grid cell
  is duplicated into `K + 1` *layers*. A move that lands on a free cell stays in the same layer; a
  move that lands on a wall steps **down one layer** (one more break used). Now every move has a
  natural cost in `{0, 1}` — `0` for a free move, `1` for breaking a wall — and we want the cheapest
  path from `(S, 0)` to any `(T, k)`.

The second framing turns the task into a single-source shortest-path problem on a layered graph whose
edge weights are all `0` or `1`. The open question is which shortest-path engine fits these weights at
this scale.

## Evaluation settings

Judged on hidden tests covering: `K = 0` (plain reachability, answer `0` or `-1`); `K = 1` (the
flip-one-wall case named in the prompt); larger `K` where the optimum breaks several walls; grids
with a free detour that beats any wall-break (answer `0`); grids where `T` is fully enclosed and the
answer equals the thickness of the surrounding wall; unreachable targets (`-1`); `1 x 2` minimal
grids; and full-size `1000 x 1000` grids with `K = 10` (so the layered graph has up to `1.1 x 10^7`
nodes) to enforce a near-linear algorithm.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int R, C, K;
    if (!(cin >> R >> C >> K)) return 0;

    vector<string> g(R);
    for (int i = 0; i < R; i++) cin >> g[i];

    int sr = -1, sc = -1, tr = -1, tc = -1;
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++) {
            if (g[i][j] == 'S') { sr = i; sc = j; }
            else if (g[i][j] == 'T') { tr = i; tc = j; }
        }

    // TODO: shortest path on the (cell x breaks_used) state graph with edge
    //       weights in {0, 1}; output the minimum number of walls broken to
    //       reach T from S, or -1 if T is unreachable with at most K breaks.
    int answer = -1;

    cout << answer << "\n";
    return 0;
}
```
