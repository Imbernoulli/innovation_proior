# LCA by Euler Tour and Sparse-Table RMQ

## Problem

For a rooted tree, preprocess once so each lowest-common-ancestor query is answered in `O(1)` time. The tree is static, so the preprocessing may spend `O(n log n)` time and space.

## Method

Run one depth-first traversal from the root and record a node every time the walk touches it: once on entry and again whenever the traversal returns to it from a child. This gives an Euler array of length `2n - 1`. Store a parallel depth array, and store `first[u]`, the first index where node `u` appears in the Euler array.

For two nodes `u` and `v`, let `left, right = sorted((first[u], first[v]))`. In the Euler slice from `left` to `right`, the traversal moves from the first node's side of the tree up through the lowest common ancestor and then down toward the second node's side. It never has to go above that ancestor, while every detour inside a child subtree is deeper. Therefore the answer is the node at the minimum-depth position in that slice:

$$
\mathrm{LCA}(u, v) =
\mathrm{euler}\left[
  \arg\min_{left \le i \le right} \mathrm{depth\_at}[i]
\right].
$$

The remaining task is static range minimum over the depth array. A sparse table stores, for each power-of-two block, the position whose depth is minimal. For a query range, choose `k = floor(log2(length))` and compare the two overlapping blocks `[left, left + 2^k - 1]` and `[right - 2^k + 1, right]`. The overlap is harmless because minimum is idempotent. This gives two table reads and one final Euler-array read.

## Code

```cpp
// Reads: "n q", then n-1 undirected tree edges (1-based), then q query pairs
// (1-based). The tree is rooted at node 0 (input node 1). Prints, one per line,
// the 1-based lowest common ancestor of each queried pair.
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    if (!(cin >> n >> q)) return 0;

    vector<vector<int>> adj(n);
    for (int i = 0; i < n - 1; ++i) {
        int u, v;
        cin >> u >> v;
        --u; --v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // One iterative DFS records the order in which vertices are touched: a
    // vertex is appended when first entered and again every time the walk
    // returns to it from a child. The tour has length 2n-1; alongside it we
    // keep the depth at each tour position and, per vertex, the index of its
    // first appearance. Iterative DFS so a path-shaped tree can't overflow.
    vector<int> euler;        // node touched at each tour position
    vector<int> depthAt;      // depth at each tour position
    vector<int> first(n, -1); // first tour position of each node
    euler.reserve(n > 0 ? 2 * n - 1 : 0);
    depthAt.reserve(n > 0 ? 2 * n - 1 : 0);

    if (n > 0) {
        vector<char> seen(n, 0);
        // frame: node, depth, next child index into adj[node]
        vector<array<int, 3>> stk;
        int root = 0;
        seen[root] = 1;
        first[root] = 0;
        euler.push_back(root);
        depthAt.push_back(0);
        stk.push_back({root, 0, 0});

        while (!stk.empty()) {
            int node = stk.back()[0];
            int d = stk.back()[1];
            int idx = stk.back()[2];
            const vector<int>& children = adj[node];
            bool advanced = false;
            while (idx < (int)children.size()) {
                int nxt = children[idx];
                ++idx;
                if (!seen[nxt]) {
                    stk.back()[2] = idx;
                    seen[nxt] = 1;
                    first[nxt] = (int)euler.size();
                    euler.push_back(nxt);
                    depthAt.push_back(d + 1);
                    stk.push_back({nxt, d + 1, 0});
                    advanced = true;
                    break;
                }
            }
            if (!advanced) {
                stk.back()[2] = idx;
                stk.pop_back();
                if (!stk.empty()) {
                    // returned to the parent: re-record it
                    euler.push_back(stk.back()[0]);
                    depthAt.push_back(stk.back()[1]);
                }
            }
        }
    }

    // Sparse table over depthAt that stores the *position* of the minimal
    // depth, so we can recover the Euler node there. min is idempotent, so
    // overlapping the two covering blocks is harmless and the query is O(1).
    int m = (int)depthAt.size();
    vector<int> logTable(m + 1, 0);
    for (int i = 2; i <= m; ++i) logTable[i] = logTable[i >> 1] + 1;
    int levels = (m > 0 ? logTable[m] + 1 : 1);

    vector<vector<int>> sp(levels, vector<int>(m, 0));
    for (int j = 0; j < m; ++j) sp[0][j] = j;
    for (int k = 1; k < levels; ++k) {
        int span = 1 << k;
        int half = 1 << (k - 1);
        for (int j = 0; j + span <= m; ++j) {
            int a = sp[k - 1][j];
            int b = sp[k - 1][j + half];
            sp[k][j] = (depthAt[a] <= depthAt[b]) ? a : b;
        }
    }

    auto argmin = [&](int l, int r) -> int {
        int k = logTable[r - l + 1];
        int a = sp[k][l];
        int b = sp[k][r - (1 << k) + 1];
        return (depthAt[a] <= depthAt[b]) ? a : b;
    };

    string out;
    out.reserve((size_t)q * 7);
    for (int i = 0; i < q; ++i) {
        int u, v;
        cin >> u >> v;
        --u; --v;
        int l = first[u], r = first[v];
        if (l > r) swap(l, r);
        int ans = euler[argmin(l, r)] + 1;
        out += to_string(ans);
        out.push_back('\n');
    }
    cout << out;
    return 0;
}
```

## Complexity

The traversal builds arrays of length `2n - 1`. The sparse table has `O(log n)` levels, so preprocessing costs `O(n log n)` time and space. Each query reads `first[u]`, `first[v]`, does one sparse-table range-minimum query over the depth array, and returns one Euler-array entry, so query time is `O(1)`.
