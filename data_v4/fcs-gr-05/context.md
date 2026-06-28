# Minimum edges to make a network bridgeless (2-edge-connectivity augmentation)

## Research question

You are given a **connected** undirected graph on `n` vertices with `m` edges (parallel edges and
self-loops are allowed). A *bridge* is an edge whose removal disconnects the graph — a single point
of failure: if that one link goes down, some part of the network becomes unreachable.

Add the **fewest** new (undirected) edges so that the resulting multigraph has **no bridge at all**,
i.e. it becomes **2-edge-connected**: it stays connected after the removal of any single edge. New
edges may join any pair of distinct vertices, including a pair that already shares an edge. Output the
minimum number of edges that must be added.

This is the bridge-connectivity augmentation problem (the 2-edge-connected case of Eswaran–Tarjan).
It is the canonical question behind making a backbone resilient to any single cable cut.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `m`
  (`1 <= n <= 2*10^5`, `0 <= m <= 2*10^5`). Each of the next `m` lines has two integers `u v`
  (`1 <= u, v <= n`) describing an undirected edge between `u` and `v`. Self-loops (`u == v`) and
  parallel edges may occur. The graph is guaranteed connected.
- Output (stdout): a single line with the minimum number of edges to add to make the graph
  2-edge-connected.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for the graph below (a triangle `1-2-3` with a pendant path branching at vertex `3`)

```
6 6
1 2
2 3
3 1
3 4
4 5
4 6
```

the answer is `2`. Edges `3-4`, `4-5`, `4-6` are bridges; contracting the 2-edge-connected pieces
leaves a tree whose leaves are `{1,2,3}`, `5`, `6` — three leaves — and `ceil(3/2) = 2`.

## Background

Two families of approach are on the table before committing to one:

- **Search over additions.** Repeatedly (or by exhaustive search) add edges between far-apart
  endpoints and re-check whether any bridge remains, stopping at the first count that succeeds.
  Re-checking bridges is `O(n + m)` via a depth-first search, but the number of candidate edge sets
  is astronomically large, so this is only viable for tiny `n`.
- **Structural / combinatorial.** Bridges partition the graph into maximal **2-edge-connected
  components** (no internal bridge). Contracting each such component to a single node turns the graph
  into a tree — the *bridge tree* — whose edges are exactly the bridges. The open questions are how to
  find the bridges at scale and what closed-form count of new edges the tree shape forces.

The scale (`n, m <= 2*10^5`) rules out any superlinear search; the intended solution is a single
linear pass to find bridges plus a closed-form count over the bridge tree.

## Evaluation settings

Judged on hidden tests covering: already-2-edge-connected graphs (answer `0`); a single bridge
(answer `1`); long paths (deep DFS, many bridges, answer `1`); star-shaped bridge trees with many
leaves (answer `ceil(L/2)` up to `10^5`); graphs full of parallel edges and self-loops (which must be
recognized as *not* bridges); a single isolated vertex (`n=1`, answer `0`); and maximum-size graphs
at `n = m = 2*10^5` for both time and recursion depth.

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

    vector<vector<pair<int,int>>> adj(n + 1);
    for (int e = 0; e < m; e++) {
        int u, v;
        cin >> u >> v;
        if (u == v) continue;            // self-loop: never a bridge
        adj[u].push_back({v, e});
        adj[v].push_back({u, e});
    }

    // TODO: find the bridges, contract 2-edge-connected components into the
    // bridge tree, and output the minimum number of edges to make it bridgeless.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
