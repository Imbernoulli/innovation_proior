# Minimum walls broken to reach the target within a budget

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
case `K = 1`.

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

## Evaluation settings

Judged on hidden tests covering: `K = 0` (plain reachability, answer `0` or `-1`); `K = 1` (the
flip-one-wall case named in the prompt); larger `K` where the optimum breaks several walls; grids
with a free detour available alongside walls to break; grids where `T` is fully enclosed by walls of
varying thickness; unreachable targets (`-1`); `1 x 2` minimal grids; and full-size `1000 x 1000`
grids with `K = 10` under the 1-second time limit.

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

    // TODO: compute the minimum number of walls broken to reach T from S while
    //       breaking at most K walls in total, or -1 if T is unreachable.
    int answer = -1;

    cout << answer << "\n";
    return 0;
}
```
