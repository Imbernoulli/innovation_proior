# Context

## Problem

Given a directed graph (flowgraph) with start vertex $s=1$, a vertex $u$ dominates $v$ if every path from $s$ to $v$ passes through $u$. For every vertex $v$ reachable from $s$, compute its immediate dominator $\mathrm{idom}(v)$ (the unique dominator of $v$, other than $v$, that is dominated by all other dominators of $v$). Output, for each vertex, the number of vertices it dominates: the subtree size of that vertex in the dominator tree, or $0$ if it is unreachable from $s$.

The graph may contain cycles, self-loops, and parallel edges, and not every vertex need be reachable from $s$ — vertices unreachable from $s$ have no dominators to report and are left out of the tree. The graph can be large ($n$ vertices and $m$ edges, each up to $\sim 10^5$ or more).

## Code framework

The deliverable is a single self-contained C++17 program reading from stdin and writing to stdout. The first line contains `n m`; the next `m` lines contain directed edges `u v` using 1-based vertex labels, and the start vertex is fixed as `s=1`. The program builds successor and predecessor adjacency lists and prints `n` integers, where the `i`th integer is the number of vertices dominated by vertex `i`, or `0` if vertex `i` is unreachable from `s`.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<vector<int>> succ(n + 1), pred(n + 1);
    for (int i = 0; i < m; ++i) {
        int u, v;
        cin >> u >> v;
        succ[u].push_back(v);
        pred[v].push_back(u);
    }

    const int s = 1;
    vector<int> ans(n + 1, 0);

    // TODO: fill ans[1..n] according to the required output.

    for (int i = 1; i <= n; ++i) {
        cout << ans[i] << " \n"[i == n];
    }
    return 0;
}
```
