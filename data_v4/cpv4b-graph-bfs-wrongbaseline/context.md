# Minimum boost energy for a gliding drone on a height grid

## Research question

A delivery drone sits on an `n x m` grid of rooftops. Cell `(i, j)` has an integer **height**
`h[i][j]`. The drone starts on the top-left rooftop `(0, 0)` and must reach the bottom-right rooftop
`(n-1, m-1)`. In one move it steps to one of the four orthogonally adjacent cells (up, down, left,
right) that lie inside the grid. The move's **energy cost** depends only on the relative heights of
the two cells:

- **Glide** — if the destination is **not higher** than the current cell (`h[dest] <= h[cur]`), the
  drone coasts there for **0** energy.
- **Boost** — if the destination is **strictly higher** (`h[dest] > h[cur]`), the drone must fire its
  rotors and pay **1** energy.

Output the **minimum total energy** to get from `(0, 0)` to `(n-1, m-1)`. The grid is always
traversable (you can reach any cell from any other by a sequence of moves), so an answer always
exists.

This is a shortest-path question on a grid graph, but the cost structure is the catch: the edge
weights are **0 or 1**, not uniform, and the weight of an edge depends on the *direction* you cross
it (going up costs 1, coming back down costs 0), so the graph is effectively directed.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `m` (`1 <= n, m <= 1000`). Then follow `n`
  lines, each with `m` integers; the `j`-th integer on the `i`-th line is `h[i][j]`
  (`0 <= h[i][j] <= 10^9`).
- Output (stdout): a single line with the minimum total boost energy from `(0,0)` to `(n-1, m-1)`.
- Time limit: 1 second. Memory: 256 MB.
- Note `n*m` can be up to `10^6`, so the algorithm must be near-linear in the number of cells.

Example. For the `3 x 3` grid

```
1 2 2
3 2 1
3 3 1
```

the answer is `1`: go down `(0,0)->(1,0)` (height `1 -> 3`, one boost, cost 1), then
`(1,0)->(2,0)->(2,1)->(2,2)` along heights `3 -> 3 -> 3 -> 1`, all glides at cost 0. Total `1`.

## Background

The grid is a graph whose vertices are cells and whose edges are the (directed) adjacency moves with
weight 0 (glide) or 1 (boost). Two shortest-path tools are on the table before committing:

- **Plain breadth-first search.** Push `(0,0)`, expand a FIFO queue, set each cell's value to its
  parent's value plus one when first discovered. BFS is `O(nm)` and the textbook tool for
  shortest paths "on a grid". The open question is *what* it minimizes when the edges are not all the
  same weight.
- **0-1 BFS.** A deque-based variant: relax a weight-0 edge by pushing the neighbour to the **front**
  of the deque and a weight-1 edge by pushing to the **back**, so the deque stays sorted by distance
  within a window of two consecutive values. Also `O(nm)`; the open question is the exact relaxation
  rule and why the front/back discipline is needed here.

## Evaluation settings

Judged on hidden tests covering: grids where the min-energy path is far longer (in moves) than the
min-moves path; flat grids (all heights equal, answer `0`); strictly increasing corridors (every step
a boost); `1 x m` and `n x 1` lines; the `1 x 1` grid; large `1000 x 1000` grids with heights up to
`10^9` (so height comparisons must not overflow and the running cost can be large); and grids where
the only cheap route doubles back. The answer is the energy, not the number of moves.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    if (!(cin >> n >> m)) return 0;
    vector<vector<int>> h(n, vector<int>(m));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            cin >> h[i][j];

    // TODO: compute the minimum total boost energy from (0,0) to (n-1,m-1),
    // where stepping to a strictly higher neighbour costs 1 and any other step costs 0.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
