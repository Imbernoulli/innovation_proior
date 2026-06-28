# Cut vertices on every path (block-cut tree queries)

## Research question

You are given an undirected graph on `n` vertices and `m` edges (it may be
disconnected, and may contain parallel edges and self-loops). Then you are given
`q` queries. Each query is a pair of vertices `u v`. For a query, a vertex `w`
(with `w != u` and `w != v`) is called **essential** for the pair if deleting `w`
from the graph (together with its incident edges) destroys *all* connection
between `u` and `v` — that is, after removing `w` there is no path from `u` to `v`.
Equivalently, `w` lies on **every** path between `u` and `v`.

For each query, report **how many** essential vertices there are. If `u` and `v`
are already in different connected components (no path exists at all), report
`-1`. If `u == v`, report `0`.

These essential vertices are exactly the **articulation points (cut vertices)**
that separate `u` from `v`. The task is the path-query version of articulation-point
analysis: not "which vertices are cut vertices globally" but "how many cut vertices
sit between *this* pair", asked many times over.

## Input / output contract

- Input (stdin):
  - The first line has two integers `n` and `m`
    (`1 <= n <= 2*10^5`, `0 <= m <= 2*10^5`).
  - Each of the next `m` lines has two integers `u v` (`1 <= u, v <= n`)
    describing an undirected edge. Self-loops (`u == v`) and parallel edges may
    appear; the solver must treat them correctly (a self-loop is never relevant to
    connectivity between two *distinct* vertices, and a pair of parallel edges is
    never a single point of failure).
  - The next line has one integer `q` (`1 <= q <= 2*10^5`).
  - Each of the next `q` lines has two integers `u v` (`1 <= u, v <= n`): a query.
- Output (stdout): for each query, one line with the number of essential vertices,
  or `-1` if `u` and `v` lie in different components, or `0` if `u == v`.
- Time limit: 2 seconds. Memory: 256 MB.

Example:

```
5 6
1 2
2 3
3 1
3 4
4 5
5 3
4
1 4
1 5
2 4
3 3
```

Output:

```
1
1
1
0
```

The graph is two triangles `{1,2,3}` and `{3,4,5}` glued at vertex `3`. For the
query `1 4`, the only vertex whose removal cuts `1` from `4` is vertex `3` (it is
the unique articulation point of the graph), so the answer is `1`; likewise for
`1 5` and `2 4`. For `3 3` the answer is `0` because `u == v`.

## Background

Two families of approach are on the table before committing to one.

- **Per-query removal test.** For a query `(u, v)`, try deleting each candidate
  vertex `w` in turn and re-run a breadth-first search to see whether `u` can still
  reach `v`; count the `w` for which it cannot. Each such test is `O(n + m)` and
  there are up to `n` candidates per query, so a single query costs `O(n(n + m))`
  and the whole input costs `O(q n (n + m))`. This is obviously correct and is the
  right tool for *checking* a fast solution, but it is hopeless at the stated scale.
- **Structural / decomposition.** Articulation points partition the graph into
  **biconnected components** (maximal subgraphs with no internal articulation
  point). Contracting that structure yields a tree-like skeleton on which "which
  cut vertices separate `u` from `v`" becomes a path question. The open questions
  are *which* skeleton makes the count exact, how to build it in linear time at
  `n, m <= 2*10^5`, and how to answer each path query in better than linear time.

The scale rules out anything superlinear per query; the intended solution builds a
linear-size decomposition once and then answers each query in `O(log n)`.

## Evaluation settings

Judged on hidden tests covering: a single vertex (`n = 1`); disconnected graphs
(some queries return `-1`); long induced paths (every interior vertex is a cut
vertex, so far-apart queries have large answers up to `~2*10^5`); large
biconnected blobs (a single cycle or clique has **no** cut vertex, every answer
`0`); graphs that mix parallel edges and self-loops (which must not be mistaken for
bridges/cut points); queries where `u` or `v` is itself a cut vertex (it must be
excluded from its own count); `u == v` queries; and maximum-size inputs at
`n = m = q = 2*10^5`, including a depth-`2*10^5` DFS path that would overflow a
recursive implementation.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    // Read the m undirected edges (self-loops and parallel edges may appear).
    for (int e = 0; e < m; e++) {
        int u, v;
        cin >> u >> v;
        // TODO: store the edge (skip / handle self-loops as appropriate).
        (void)u; (void)v;
    }

    // TODO: decompose the graph into biconnected components and articulation
    //       points; build the block-cut tree; preprocess it for path queries.

    int q;
    cin >> q;
    while (q--) {
        int u, v;
        cin >> u >> v;
        // TODO: output the number of cut vertices on every u-v path
        //       (or -1 if u, v are in different components, 0 if u == v).
        long long answer = 0;
        cout << answer << "\n";
    }
    return 0;
}
```
