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
for *all* `n` roots fast enough that the per-node work amortizes to a constant, which is the kind of
"reroot the tree and reuse the parent's answer" pattern that recurs across tree-DP problems.

## Input / output contract

- Input (stdin): the first token is `n` (`1 <= n <= 2*10^5`). Then `n-1` lines follow, each with two
  integers `u v` (`1 <= u, v <= n`, `u != v`), describing an undirected edge. The edges are
  guaranteed to form a tree (connected, acyclic). Endpoints may appear in either order.
- Output (stdout): `n` lines; line `v` contains `S(v)`, the sum of distances from node `v` to every
  node.
- Time limit: 1 second. Memory: 256 MB.

Example: for the path `1 - 2 - 3 - 4` (edges `1 2`, `2 3`, `3 4`) the answer is
`S(1)=6, S(2)=4, S(3)=4, S(4)=6`.

## Background

The single-source quantity is easy: one traversal from a fixed root gives every node's depth, and the
sum of depths is `S(root)`. The whole difficulty is the *all-roots* requirement, and two families of
approach are on the table before committing to one:

- **BFS/DFS from each node.** Run a traversal rooted at each of the `n` nodes and sum the depths each
  time. Each traversal is `O(n)`, so the total is `O(n^2)`. Simple and obviously correct; the open
  question is whether `O(n^2)` is fast enough at the stated scale.
- **Rerooting (two-pass tree DP).** Compute the answer for one fixed root from subtree sizes, then
  derive every other node's answer from its parent's answer by a single transfer step as the root
  "moves" along an edge. This is `O(n)`; the open question is the exact transfer formula and the
  bookkeeping (subtree sizes, traversal order) that makes it correct.

## Evaluation settings

Judged on hidden tests covering: tiny trees (`n = 1`, `n = 2`), long paths (which maximize depth and
push any recursive traversal toward stack overflow, and make `S` of the endpoints as large as
`n(n-1)/2 ~ 2*10^10`, exceeding 32-bit range), stars (one center, `n-1` leaves), balanced and random
trees, and the largest case `n = 2*10^5`. Edges are given in arbitrary order and orientation.

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
