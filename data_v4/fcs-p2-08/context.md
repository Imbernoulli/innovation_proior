# Longest path (by number of edges) from a source in a DAG

## Research question

You are given a directed acyclic graph (DAG) with `V` vertices (numbered `1..V`) and `E` directed
edges. A *source* is a vertex with in-degree `0`. Consider every directed path that **starts at a
source** and follows edges forward; because the graph is acyclic, every such walk is automatically a
simple path and is finite. Output the **maximum number of edges** on any such path (a single isolated
vertex is a path with `0` edges, so the answer is always at least `0`).

This "longest chain length" quantity is the backbone of critical-path / scheduling computations
(the longest dependency chain in a build graph, the critical path in a task DAG), so getting the
length exactly right — including the multi-source, multi-sink, and no-edge corners — matters.

## Input / output contract

- Input (stdin): the first line has two integers `V` and `E`
  (`1 <= V <= 2*10^5`, `0 <= E <= 4*10^5`). Each of the next `E` lines has two integers `u v`
  (`1 <= u, v <= V`) denoting a directed edge `u -> v`. The graph is guaranteed acyclic. Parallel
  edges may appear; there are no self-loops.
- Output (stdout): a single line with the maximum number of edges on a path that starts at a source.
- Time limit: 1 second. Memory: 256 MB.

Example: for the graph below with `V = 9`

```
9 8
1 2
2 3
2 4
2 5
1 6
6 7
7 8
8 9
```

the answer is `4`: the path `1 -> 6 -> 7 -> 8 -> 9` uses `4` edges. (Vertex `2` has the highest
out-degree, but `1 -> 2 -> {3,4,5}` dead-ends after only `2` edges.)

## Background

The phrase "longest path" is NP-hard for general graphs, but on a **DAG** it is polynomial, and two
families of approach are on the table before committing to one:

- **Greedy descent.** Start at each source and, at every step, walk to the out-neighbour that "looks
  most promising" — for instance the neighbour with the largest out-degree (the most options ahead).
  This is `O(V + E)` and a few lines. The open question is whether a local choice can be trusted to
  produce a globally longest path under the forward-reachability structure.
- **Topological-order dynamic programming.** Process vertices in topological order and carry, for each
  vertex `v`, the longest path (in edges) that ends at `v`. This is `O(V + E)`; the open question is
  the exact recurrence, the base case for sources, and what the global answer reads off.

## Evaluation settings

Judged on hidden tests covering: graphs with no edges (`E = 0`, answer `0`); a single long chain;
"wide hub then long tail" graphs where a high-out-degree neighbour leads to a short dead-end while a
low-out-degree neighbour starts a long chain; multi-source / multi-sink layered DAGs; graphs with
parallel edges; and large instances with `V = 2*10^5`, `E = 4*10^5` where an `O(V*E)` method would be
too slow.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;
    vector<vector<int>> adj(n + 1);        // 1-indexed; adj[u] = out-neighbours
    vector<int> indeg(n + 1, 0);
    for (int e = 0; e < m; e++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        indeg[v]++;
    }

    // TODO: output the maximum number of edges on a path that starts at a
    // source (an in-degree-0 vertex) of this DAG.
    int answer = 0;

    cout << answer << "\n";
    return 0;
}
```
