# Sum of distances from every node in a tree

## Research question

You are given an undirected tree with `n` nodes (numbered `1..n`) and `n-1` edges. For **every** node
`v`, define `S(v)` as the sum of the shortest-path distances from `v` to all other nodes:

```
S(v) = sum over all u of dist(v, u)
```

Output `S(1), S(2), ..., S(n)`. The distance between two nodes in a tree is the number of edges on
the unique path connecting them.

This is the all-nodes version of a classic single-source quantity. The single-source value `S(v)` for
one fixed `v` is a one-pass BFS/DFS; the question that makes this interesting is producing the value
for *all* `n` roots fast enough at the stated scale.

## Input / output contract

- Input (stdin): the first token is `n` (`1 <= n <= 2*10^5`). Then `n-1` lines follow, each with two
  integers `u v` (`1 <= u, v <= n`, `u != v`), describing an undirected edge. The edges are
  guaranteed to form a tree (connected, acyclic). Endpoints may appear in either order.
- Output (stdout): `n` lines; line `v` contains `S(v)`, the sum of distances from node `v` to every
  node.
- Time limit: 1 second. Memory: 256 MB.

Example: for the path `1 - 2 - 3 - 4` (edges `1 2`, `2 3`, `3 4`) the answer is
`S(1)=6, S(2)=4, S(3)=4, S(4)=6`.

## Evaluation settings

Judged on hidden tests covering: tiny trees (`n = 1`, `n = 2`), long paths (which maximize depth and
push any recursive traversal toward stack overflow), stars (one center, `n-1` leaves), balanced and
random trees, and the largest case `n = 2*10^5`. Edges are given in arbitrary order and orientation.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;

    vector<vector<int>> adj(n + 1);
    for (int e = 0; e < n - 1; e++) {
        int u, v;
        cin >> u >> v;            // 1-indexed endpoints of an undirected edge
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // TODO: for every node v, compute S(v) = sum of distances from v to all nodes,
    // and print S(1)..S(n), one per line.

    return 0;
}
```
