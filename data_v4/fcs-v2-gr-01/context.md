# Min-cost flow of a fixed value with negative edge costs

## Research question

You are given a directed graph with `n` vertices and `m` edges. Each edge `u -> v` has a non-negative
integer capacity `cap` and an integer cost `cost` **that may be negative**. You must route exactly `F`
units of flow from a source `s` to a sink `t` at **minimum total cost**, where the cost of routing
`x` units along an edge is `x * cost`. If it is impossible to push `F` units from `s` to `t`, report
that.

The graph is guaranteed to contain **no negative-cost cycle** (so the minimum-cost flow is well
defined and finite).

This is the core min-cost-flow primitive behind assignment, transportation, and project-selection
problems. Getting it right when some costs are negative is the point of the exercise.

## Input / output contract

- Input (stdin):
  - First line: five integers `n m s t F`
    (`2 <= n <= 1000`, `0 <= m <= 10000`, `0 <= s,t < n`, `s != t`, `0 <= F <= 10^9`).
  - Next `m` lines: four integers `u v cap cost` describing a directed edge `u -> v`
    (`0 <= u,v < n`, `u != v`, `0 <= cap <= 10^9`, `-10^6 <= cost <= 10^6`).
  - Multiple (parallel) edges between the same ordered pair are allowed.
  - The graph contains no negative-cost cycle.
- Output (stdout): a single line. If `F` units can be routed from `s` to `t`, print the minimum total
  cost (a possibly-negative integer). Otherwise print `IMPOSSIBLE`.
- Time limit: 2 seconds. Memory: 256 MB.

Example: with the bipartite assignment graph

```
6 8 0 5 2
0 1 1 0
0 2 1 0
1 3 1 4
1 4 1 -2
2 3 1 3
2 4 1 1
3 5 1 0
4 5 1 0
```

the answer is `1`: route worker `1 -> task 4` (cost `-2`) and worker `2 -> task 3` (cost `3`), total
`-2 + 3 = 1`. The alternative pairing `1 -> 3`, `2 -> 4` costs `4 + 1 = 5`, so `1` is optimal.

## Evaluation settings

Judged on hidden tests covering: graphs with several negative edges; the empty / no-edge graph;
`F = 0` (answer `0`); `F` larger than the max flow (`IMPOSSIBLE`); parallel edges with different costs;
long negative chains; zero-capacity edges; and large instances at `n = 1000`, `m = 10^4`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, s, t;
    long long F;
    if (!(cin >> n >> m >> s >> t >> F)) return 0;

    // Read m directed edges u->v with capacity cap and (possibly negative) cost.
    for (int i = 0; i < m; i++) {
        int u, v; long long cap, cost;
        cin >> u >> v >> cap >> cost;
        // TODO: store the edge (and its residual reverse arc).
    }

    // TODO: route exactly F units s->t at minimum cost; print the cost or IMPOSSIBLE.

    return 0;
}
```
