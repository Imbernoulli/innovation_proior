# Context

## Problem

Given a directed acyclic graph on $n$ vertices, find the minimum number of
vertex-disjoint paths needed to cover every vertex exactly once (the minimum path
cover). A path here is a directed simple path that follows edges of the graph; a
single vertex with no edges counts as a path of length zero. "Vertex-disjoint" and
"exactly once" mean the chosen paths partition the vertex set: every vertex lies on
one and only one path. Make the number of paths as small as possible.

The graph is acyclic, so the edge relation is consistent with some topological
order, and no path can revisit a vertex. Sizes are moderate: $n$ up to a few
thousand vertices and $m$ up to tens of thousands of edges, so an $O(n \cdot m)$
procedure is acceptable.

## Code framework

The deliverable is a single self-contained C++17 program. It reads `n m`
followed by `m` directed edges `u v` from stdin, with edge endpoints 1-based in
the input, stores them as a 0-based adjacency list, and writes one integer to
stdout: the minimum number of vertex-disjoint paths covering every vertex
exactly once.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long m;
    if (!(cin >> n >> m)) return 0;

    vector<vector<int>> adj(n);
    for (long long i = 0; i < m; ++i) {
        int u, v;
        cin >> u >> v;
        --u;
        --v;
        adj[u].push_back(v);
    }

    long long answer = 0;

    // TODO:

    cout << answer << "\n";
    return 0;
}
```
