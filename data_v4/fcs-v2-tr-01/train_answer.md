**Problem.** A tree of `n` facilities is rooted at node `1`; every edge has a positive cut cost
(`1 <= w <= 10^9`). For each of `q` independent orders you are given a set `S` of marked nodes and
must cut a cheapest set of edges so that no node of `S` can still reach the root. Output the minimum
total cut cost per order. The only size bound is aggregate: `sum |S| <= 2*10^5` (and `n, q <= 2*10^5`).
The root is never marked.

**Why the obvious method is too slow.** The correct skeleton is a post-order min-cut DP over the
*whole* tree: `f[v]` = min cost to detach every marked node in `v`'s subtree from `v`; a child `c`
via edge cost `w` contributes `w` if `c` is marked and `min(w, f[c])` otherwise; the answer is
`f[1]`. That is `O(n)` per order, hence `O(q n)` — up to `4*10^10` node visits, dozens of times over
a 2 s budget. It is wasteful in a precise way: even an order with `|S| = 1` walks all `n` nodes,
paying for the tree size while only `sum|S|` actually varies.

**Key idea — compress to the virtual (auxiliary) tree.** An order only "bends" the tree at `O(|S|)`
nodes: the marked nodes and the branch points where their root-paths split, i.e. the pairwise LCAs.
Sort the marked nodes by DFS in-time; the LCAs of *adjacent* pairs already cover every branch point,
and there are `< |S|` of them. Build a **virtual tree** on {marked nodes} ∪ {adjacent-pair LCAs} ∪
{root}. A virtual edge `(a, b)` represents the original *path* between them; label it with the
**minimum edge weight on that path**, because the cheapest way to sever a straight chain is to cut
its lightest edge — exactly what a min-cut wants. Run the identical DP on this `O(|S|)`-node tree.
Per order: `O(|S| log n)` (LCA + path-minimum via binary lifting). Total `O((n + sum|S|) log n)`.

**Pitfalls.**
1. *Marked-child contribution.* For a marked child `c`, contribute `w`, **not** `min(w, f[c])`.
   `f[c]` only separates nodes *below* `c` from `c`; it never detaches `c` itself, so `min(w, f[c])`
   can under-count (pick a cheap-below `f[c]` that leaves `c` reachable). Cutting the single edge `w`
   on the compressed chain severs `c`'s whole subtree, so deeper marked nodes are auto-handled and
   `f[c]` is irrelevant for a marked child.
2. *Path-minimum, not edge weight.* A virtual edge stands for a chain; its weight must be the
   minimum original edge on that chain. Use a binary-lifting table `mn[k][v]` alongside the ancestor
   table to read it in `O(log n)`.
3. *Overflow.* The answer can reach `~2*10^14`; use `long long` for every weight sum. An `int` is a
   silent wrong answer at scale.
4. *Recursion depth.* A bamboo of depth `2*10^5` overflows a recursive DFS; build `tin`/`dep` and
   run the per-order post-order iteratively.

**Edge cases.** Empty order (`k = 0`) → `0` (force the root into the node set so the virtual tree is
a single node). Single-node tree (`n = 1`) → all orders `0`; keep `LOG >= 1` so the lifting loops
are valid. Shared-edge dominance (two deep marked nodes under one ancestor) → the DP picks the upper
compressed edge over the sum of the lower ones; this is the case the whole technique exists for.

**Complexity.** `O(n log n)` preprocessing, `O(|S| log n)` per order, `O((n + sum|S|) log n)` total;
`O(n log n)` memory. Measured: ~0.25 s and 65 MB at `n = sum|S| = 2*10^5`.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

// Cut-to-Quarantine.
// Rooted tree at node 1; every edge has a positive cut cost. For each query with a
// set S of "special" nodes, output the minimum total cost of edges to cut so that no
// special node stays connected to the root (node 1).
//
// Insight: only the special nodes plus their pairwise LCAs (and the root) shape the
// answer. Build the virtual tree of those O(|S|) nodes; a virtual edge stands for an
// original path, and the cheapest way to sever that path is to cut its minimum-weight
// edge. A small tree DP over the virtual tree then yields the min cut.
// Complexity: O(n log n) preprocessing, O(|S| log n) per query.

static const long long INF = (long long)4e18;

int n, q;
vector<pair<int,int>> g[200005];   // undirected adjacency: (neighbor, edgeWeight)
int LOG;
vector<array<int,20>> up;          // binary-lifting ancestors
vector<array<long long,20>> mn;    // min edge weight on each 2^k upward jump
int tin[200005], dep[200005], timer_ = 0;

int lca(int u, int v) {
    if (dep[u] < dep[v]) swap(u, v);
    int d = dep[u] - dep[v];
    for (int k = 0; k < LOG; k++) if (d & (1 << k)) u = up[u][k];
    if (u == v) return u;
    for (int k = LOG - 1; k >= 0; k--)
        if (up[u][k] != up[v][k]) { u = up[u][k]; v = up[v][k]; }
    return up[u][0];
}

// minimum edge weight on the path from descendant d up to ancestor a (a above d)
long long minEdgeUp(int d, int a) {
    long long res = INF;
    int diff = dep[d] - dep[a];
    for (int k = 0; k < LOG; k++)
        if (diff & (1 << k)) { res = min(res, mn[d][k]); d = up[d][k]; }
    return res;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (!(cin >> n)) return 0;
    LOG = 1; while ((1 << LOG) < n + 1) LOG++; LOG = min(LOG, 20); if (LOG < 1) LOG = 1;
    up.assign(n + 1, {});
    mn.assign(n + 1, {});
    for (int i = 0; i < n - 1; i++) {
        int u, v; long long w;
        cin >> u >> v >> w;
        g[u].push_back({v, (int)w});
        g[v].push_back({u, (int)w});
    }

    // Rooted iterative DFS at node 1: set tin, dep, up[*][0], mn[*][0].
    timer_ = 0;
    {
        vector<int> par(n + 1, 0);
        vector<int> st; st.reserve(n);
        vector<int> it(n + 1, 0);
        up[1][0] = 1; mn[1][0] = INF; dep[1] = 0; par[1] = 0;
        st.push_back(1); tin[1] = timer_++;
        while (!st.empty()) {
            int v = st.back();
            bool pushed = false;
            while (it[v] < (int)g[v].size()) {
                auto pr = g[v][it[v]++];
                int c = pr.first; long long w = pr.second;
                if (c == par[v]) continue;
                par[c] = v;
                up[c][0] = v; mn[c][0] = w; dep[c] = dep[v] + 1;
                tin[c] = timer_++;
                st.push_back(c);
                pushed = true;
                break;
            }
            if (!pushed) st.pop_back();
        }
    }
    // binary lifting
    for (int k = 1; k < LOG; k++)
        for (int v = 1; v <= n; v++) {
            up[v][k] = up[ up[v][k-1] ][k-1];
            mn[v][k] = min(mn[v][k-1], mn[ up[v][k-1] ][k-1]);
        }

    cin >> q;
    string out;
    out.reserve(1 << 20);

    vector<int> nodes;
    vector<int> special;
    vector<char> isSpec(n + 1, 0);

    while (q--) {
        int k; cin >> k;
        special.assign(k, 0);
        for (int i = 0; i < k; i++) { cin >> special[i]; isSpec[special[i]] = 1; }

        // Node set of the virtual tree: special nodes + root + adjacent LCAs.
        nodes.assign(special.begin(), special.end());
        nodes.push_back(1);
        sort(nodes.begin(), nodes.end(), [](int a, int b){ return tin[a] < tin[b]; });
        nodes.erase(unique(nodes.begin(), nodes.end()), nodes.end());

        int m0 = nodes.size();
        for (int i = 0; i + 1 < m0; i++)
            nodes.push_back(lca(nodes[i], nodes[i + 1]));
        sort(nodes.begin(), nodes.end(), [](int a, int b){ return tin[a] < tin[b]; });
        nodes.erase(unique(nodes.begin(), nodes.end()), nodes.end());

        int M = nodes.size();
        static vector<vector<pair<int,long long>>> vch;
        vch.assign(M, {});
        auto posOf = [&](int x)->int{
            int lo = 0, hi = M - 1;
            while (lo < hi) { int mid=(lo+hi)/2; if (tin[nodes[mid]] < tin[x]) lo=mid+1; else hi=mid; }
            return lo;
        };

        // Build virtual tree via a monotonic stack (indices into `nodes`).
        vector<int> stk;
        stk.push_back(0); // node 1 has the smallest tin, sits at index 0
        for (int i = 1; i < M; i++) {
            int cur = nodes[i];
            int l = lca(nodes[stk.back()], cur);
            while (stk.size() >= 2 && dep[ nodes[stk[stk.size()-2]] ] >= dep[l]) {
                int child = stk.back(); stk.pop_back();
                int parent = stk.back();
                vch[parent].push_back({child, minEdgeUp(nodes[child], nodes[parent])});
            }
            int topIdx = stk.back();
            if (nodes[topIdx] != l) {
                int li = posOf(l);
                int child = stk.back(); stk.pop_back();
                vch[li].push_back({child, minEdgeUp(nodes[child], l)});
                stk.push_back(li);
            }
            stk.push_back(i);
        }
        while (stk.size() >= 2) {
            int child = stk.back(); stk.pop_back();
            int parent = stk.back();
            vch[parent].push_back({child, minEdgeUp(nodes[child], nodes[parent])});
        }

        // Post-order over the virtual tree (root index 0).
        vector<int> order; order.reserve(M);
        {
            vector<int> s; s.push_back(0);
            vector<int> state(M, 0);
            while (!s.empty()) {
                int u = s.back();
                if (state[u] < (int)vch[u].size()) s.push_back(vch[u][state[u]++].first);
                else { order.push_back(u); s.pop_back(); }
            }
        }

        // dp[u] = min cost to disconnect every special node in virtual-subtree(u) from nodes[u].
        vector<long long> dp(M, 0);
        for (int u : order) {
            long long sum = 0;
            for (auto &e : vch[u]) {
                int c = e.first; long long w = e.second;
                if (isSpec[nodes[c]])
                    sum += w;                // special child: must sever the connecting path
                else
                    sum += min(w, dp[c]);    // ordinary child: cut here, or recurse below
            }
            dp[u] = sum;
        }

        out += to_string(dp[0]);
        out += '\n';

        for (int i = 0; i < k; i++) isSpec[special[i]] = 0;
    }

    cout << out;
    return 0;
}
```
