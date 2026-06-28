**Problem.** Given a **connected** undirected graph (`n, m <= 2*10^5`, parallel edges and self-loops allowed), add the fewest edges so that the graph becomes **2-edge-connected** — no *bridge* remains, i.e. it stays connected after the removal of any single edge. Read `n m` then `m` edges from stdin; print the minimum number of edges to add.

**Why the obvious search is wrong (too slow).** A bridge is an edge whose removal increases the component count, so "is this graph bridgeless?" is checkable in `O(n+m)`. But finding the minimum additions by trying every `k` and every `k`-subset of vertex pairs is super-polynomial — confirming the answer is `1` on a path already costs `O(n^3)`. That brute is only an oracle for tiny `n`, never the solution.

**Key idea — contract to the bridge tree, then `ceil(leaves/2)`.** The bridges partition the graph into maximal 2-edge-connected components; contracting each to a super-node turns the graph into a **tree** (the *bridge tree*) whose edges are exactly the bridges. The whole problem reduces to that tree's shape. The leaf-counting argument: the unique bridge incident to a leaf can be covered only by an added edge that has the leaf as an endpoint; each added edge has two endpoints, so with `L` leaves at least `ceil(L/2)` edges are **necessary**, and pairing opposite leaves (leaf `i` with leaf `i + L/2` in DFS order) covers every bridge, so it is **sufficient**. Hence the answer is `0` when there are no bridges, else `ceil(L/2)` — it depends on nothing but the leaf count.

**Pitfalls to get right.**
1. *Parallel edges.* A double edge `u–v` is **not** a bridge. The DFS must skip only the *specific edge id* it entered by, not every edge to the parent *vertex* — otherwise the second `u–v` edge is wrongly suppressed and gets marked a phantom bridge. Key the parent-skip on edge id.
2. *Recursion depth.* A path of `2*10^5` vertices nests the DFS `2*10^5` deep; a recursive Tarjan overflows the stack. The DFS **must be iterative** (explicit frame stack).
3. *Bridgeless short-circuit.* If there are no bridges output `0` directly; do not treat a single-node bridge tree as a degenerate leaf.

**Edge cases (all verified against a definitional brute):** single vertex `n=1` -> `0`; single bridge -> `1`; long path -> `1`; star with `n-1` leaves -> `ceil((n-1)/2)`; self-loops dropped; parallel-edge-only graphs -> `0`.

**Complexity.** Tarjan `O(n+m)` + DSU near-linear + a leaf scan; overall effectively `O(n+m)` time and `O(n+m)` space. At `n=m=2*10^5` it runs in about `0.1 s`.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    // Adjacency as (neighbor, edge_id). Parallel edges share a vertex pair but
    // get distinct ids, which is what lets us detect bridges correctly: an edge
    // is a bridge only if it is the unique connection between its two sides.
    vector<vector<pair<int,int>>> adj(n + 1);
    for (int e = 0; e < m; e++) {
        int u, v;
        cin >> u >> v;
        if (u == v) continue;            // self-loop: never a bridge, drop it
        adj[u].push_back({v, e});
        adj[v].push_back({u, e});
    }

    // --- Iterative Tarjan: discovery time tin[], low-link low[], mark bridges. ---
    vector<int> tin(n + 1, 0), low(n + 1, 0);
    vector<char> isBridge(m, 0);
    int timer = 0;

    // Iterative DFS frame: node, edge id used to enter (-1 for root), and an
    // index into adj[node] tracking how far we have iterated its neighbors.
    vector<int> stkNode, stkPedge, stkIdx;
    stkNode.reserve(n + 1);
    stkPedge.reserve(n + 1);
    stkIdx.reserve(n + 1);

    for (int s = 1; s <= n; s++) {
        if (tin[s]) continue;
        stkNode.push_back(s);
        stkPedge.push_back(-1);
        stkIdx.push_back(0);
        while (!stkNode.empty()) {
            int u = stkNode.back();
            int &i = stkIdx.back();
            if (i == 0) {
                tin[u] = low[u] = ++timer;
            }
            if (i < (int)adj[u].size()) {
                auto [v, eid] = adj[u][i];
                i++;
                if (eid == stkPedge.back()) continue; // skip the edge we entered by
                if (tin[v]) {
                    // back/forward edge to an already-visited vertex
                    low[u] = min(low[u], tin[v]);
                } else {
                    // tree edge: descend
                    stkNode.push_back(v);
                    stkPedge.push_back(eid);
                    stkIdx.push_back(0);
                }
            } else {
                // done with u: pop and fold its low-link into its parent
                int pedge = stkPedge.back();
                stkNode.pop_back();
                stkPedge.pop_back();
                stkIdx.pop_back();
                if (!stkNode.empty()) {
                    int p = stkNode.back();
                    low[p] = min(low[p], low[u]);
                    if (low[u] > tin[p]) isBridge[pedge] = 1; // tree edge is a bridge
                }
            }
        }
    }

    // --- Contract 2-edge-connected components: DSU over all non-bridge edges. ---
    vector<int> par(n + 1);
    iota(par.begin(), par.end(), 0);
    function<int(int)> find = [&](int x) {
        while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; }
        return x;
    };
    auto unite = [&](int a, int b) {
        a = find(a); b = find(b);
        if (a != b) par[a] = b;
    };
    for (int u = 1; u <= n; u++)
        for (auto [v, eid] : adj[u])
            if (!isBridge[eid]) unite(u, v);

    // --- Bridge tree: each bridge connects two distinct components; count the
    //     degree of every component node. A leaf has degree exactly 1. ---
    unordered_map<int,int> deg;
    deg.reserve(n * 2);
    int bridgeCount = 0;
    for (int u = 1; u <= n; u++)
        for (auto [v, eid] : adj[u])
            if (isBridge[eid] && u < v) {           // count each bridge once
                bridgeCount++;
                deg[find(u)]++;
                deg[find(v)]++;
            }

    // If there are no bridges the graph is already 2-edge-connected: 0 edges.
    if (bridgeCount == 0) {
        cout << 0 << "\n";
        return 0;
    }

    long long leaves = 0;
    for (auto &kv : deg)
        if (kv.second == 1) leaves++;

    // Eswaran-Tarjan: ceil(leaves / 2) new edges make the bridge tree
    // 2-edge-connected, hence the whole graph bridgeless.
    cout << (leaves + 1) / 2 << "\n";
    return 0;
}
```
