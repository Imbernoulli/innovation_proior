**Problem.** Given an undirected tree on `n` nodes (`1 <= n <= 2*10^5`), output for every node `v` the value `S(v) = sum over all u of dist(v, u)`, the total distance from `v` to all other nodes. Read `n` and the `n-1` edges from stdin; print `S(1)..S(n)`, one per line.

**Why the obvious approach is too slow.** Single-source is easy — one traversal from a fixed root gives every depth, and the sum of depths is `S(root)`. The naive all-nodes algorithm just does that from each of the `n` roots: `O(n)` per root, `O(n^2)` total. At `n = 2*10^5` that's `4*10^{10}` operations — roughly four orders of magnitude past a 1-second budget, not a constant I can tune away. I need `O(1)` amortized per node, i.e. `O(n)` total, which forces *reusing* a neighbour's answer instead of recomputing.

**Key idea — two-pass rerooting with the transfer formula.** Root the tree at node `1` and let `size[v]` be the subtree size of `v`. The single transfer step is the whole trick: when the root slides from a node `par` to an adjacent `child`, the edge `par–child` splits the tree in two. Every node *inside* `child`'s subtree (`size[child]` of them) gets one step **closer**; every node *outside* (`n - size[child]` of them) gets one step **farther**. Net change:

```
S(child) = S(par) + (n - 2 * size[child]).
```

So compute `S(1)` (sum of depths) and all subtree sizes in one DFS, then propagate the answer outward in discovery order — each child read in `O(1)` from its already-known parent. Total `O(n)`. Sanity check the sign: if `child`'s subtree is more than half the tree, `n - 2*size[child] < 0` and the sum drops (we move toward the bulk of the nodes) — correct.

**Pitfalls to get right.**
1. *Iterative traversal, not recursion.* The worst case is a path of `2*10^5` nodes — recursion depth `2*10^5` overflows the default stack. Both passes use an explicit stack / discovery-order array.
2. *DFS visited-guard.* Guard re-visits with an explicit `visited` flag set **at push time**, not the fragile `w != parent[u]` test; this provably enters each node into `order` exactly once, with parents strictly before children (which the reroot pass relies on).
3. *Overflow.* A path endpoint sees `S = n(n-1)/2 ~ 2*10^{10}`, well past 32-bit range. All sums, sizes, and the answer are `long long`, and `n` is cast to `long long` inside the formula so `n - 2*size[u]` is 64-bit.
4. *Order of the reroot pass.* Process nodes parent-before-child (discovery order) so `S(par)` is ready when computing `S(child)`.

**Edge cases.** `n = 1` -> print `0` (special-cased; no edges, distance to self is 0). `n = 2` -> both nodes print `1`. Long path -> iterative DFS survives, 64-bit sums hold. Star -> center `n-1`, each leaf `1 + 2*(n-2)`; the formula reproduces both.

**Complexity.** `O(n)` time, `O(n)` space (adjacency + a handful of length-`n` arrays).

**Code.**

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

    if (n == 1) {                 // single node: distance to everyone (itself) is 0
        cout << 0 << "\n";
        return 0;
    }

    // ---- Pass 1: iterative DFS from root 1 to get parent order, subtree sizes,
    //              and S(root) = sum of depths over all nodes. ----
    vector<int> parent(n + 1, 0);
    vector<int> order;            // nodes in DFS discovery order (root first)
    order.reserve(n);
    vector<long long> size(n + 1, 1);
    vector<long long> ans(n + 1, 0);

    {
        vector<int> st;
        st.reserve(n);
        st.push_back(1);
        parent[1] = 0;
        vector<char> visited(n + 1, 0);
        visited[1] = 1;
        while (!st.empty()) {
            int u = st.back();
            st.pop_back();
            order.push_back(u);
            for (int w : adj[u]) {
                if (!visited[w]) {
                    visited[w] = 1;
                    parent[w] = u;
                    st.push_back(w);
                }
            }
        }
    }

    // depth of root is 0; depth[child] = depth[parent] + 1, processed in discovery order.
    vector<long long> depth(n + 1, 0);
    long long rootSum = 0;
    for (int idx = 0; idx < (int)order.size(); idx++) {
        int u = order[idx];
        if (u != 1) depth[u] = depth[parent[u]] + 1;
        rootSum += depth[u];
    }

    // subtree sizes: process discovery order in reverse so children precede parents.
    for (int idx = (int)order.size() - 1; idx >= 1; idx--) {
        int u = order[idx];
        size[parent[u]] += size[u];
    }

    ans[1] = rootSum;

    // ---- Pass 2: reroot in discovery order (parent computed before child). ----
    // Moving the root from par to child: the size[child] nodes in child's subtree
    // get one step closer, the other (n - size[child]) get one step farther.
    for (int idx = 1; idx < (int)order.size(); idx++) {
        int u = order[idx];
        int p = parent[u];
        ans[u] = ans[p] + (long long)n - 2LL * size[u];
    }

    string out;
    out.reserve((size_t)n * 12);
    for (int v = 1; v <= n; v++) {
        out += to_string(ans[v]);
        out += '\n';
    }
    cout << out;
    return 0;
}
```
