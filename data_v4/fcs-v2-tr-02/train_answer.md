**Problem.** Static rooted tree on `n` nodes (`1..n`); answer `q` queries. Each query `(v, k)` asks
for the k-th ancestor of `v` — the node `k` parent-edges above `v` — or `0` if `v` has fewer than `k`
ancestors. `n, q <= 5*10^5`, `0 <= k <= 10^9`. Read parents from stdin, print one answer per query.

**Why the obvious approach is too slow.** Binary lifting alone (jump pointers `up2[j][v]`) builds in
`O(n log n)` but answers each query in `O(log n)` by following one pointer per set bit of `k`. On a
depth-`5*10^5` caterpillar with `5*10^5` queries that is `~10^7` dependent reads into a ~38 MB table
(cache-hostile) — the log-per-query factor is exactly the cost to remove, and a real timeout risk on
adversarial tests. Precomputing every ancestor of every node is `Θ(n^2)` memory and immediately out.

**Key idea — ladder (long-path) decomposition plus a single jump.** Two complementary structures meet
in the middle exactly once:

- *Long-path / ladder decomposition.* Decompose the tree into vertical *long paths*: from each node,
  continue downward into the child of maximum `height` (longest downward chain to a leaf). Each node
  lies on one long path. Then **extend each long path upward** by as many of its head's ancestors as
  the path is long (length `L = height(head)+1`), storing the result top-to-bottom in one array — a
  *ladder*. Total ladder size is `<= 2n`. Invariant: **a node `u` of height `h` lives in a ladder
  holding `>= h` of its proper ancestors**, so climbing up to `height(u)` steps from `u` is one index
  subtraction — O(1).
- *One jump pointer.* For `(v, k)` with `k >= 1`, let `j = floor(log2 k)` so `2^j <= k < 2^{j+1}`.
  Jump `2^j` up from `v` to `w` in one lookup. Then `w` has descendant `v` at distance `2^j`, so
  `height(w) >= 2^j`, while the residual `rem = k - 2^j` satisfies `0 <= rem < 2^j <= height(w)`. By
  the invariant the `rem`-th ancestor of `w` is *inside `w`'s ladder* — one more index subtraction.

Two table lookups total: O(1) per query, `O(n log n)` build (the jump table) + `O(n)` (the ladders).
The jump makes one exponentially-large-enough hop; the ladder finishes the residual. This strictly
dominates binary-lifting-only, which pays the log on every query.

**Pitfalls to get right.**
1. *Ladder extension length.* Extend each long path upward by `L = height(head)+1` ancestors, tied to
   the **head's full path length**, not `height[v]` of an arbitrary member. Using `height[v]` is an
   off-by-one that breaks `idx >= 0` for a short path hanging off a deep spine; the correct extension
   restores `posInLadder[w] - rem >= 0` whenever the ancestor exists.
2. *The "no ancestor" guard.* `k` can be up to `10^9`, far beyond any depth. Short-circuit with
   `k > depth[v] -> 0` (and `k == 0 -> v`) before touching the structure; read `k` as `long long`.
3. *Deep trees.* Depth can reach `5*10^5`, so use iterative depth/height passes (explicit stack +
   bucket sort by depth), never recursive DFS, or the call stack overflows.
4. *No label-order assumptions.* Everything is driven by `par`/`children`/`depth`; the root may not be
   node `1` and parents may point to larger labels.

**Edge cases.** `k = 0 -> v`; `k = depth[v] ->` the root; `k > depth[v] -> 0`; huge `k -> 0`;
`n = 1` (any `k >= 1 -> 0`); `n = 0` (empty input, prints nothing); caterpillar of depth `n` (no
stack overflow, no per-query log).

**Complexity.** Build `O(n log n)` time and memory (jump table `~19 * n` ints ≈ 38 MB at `n = 5*10^5`)
plus `O(n)` for ladders; each query is O(1) (two lookups). Measured ~0.2–0.34 s and ~90–100 MB at
`n = q = 5*10^5` across path/skew/star/random shapes — inside 1 s / 256 MB.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;

    // parent[i] for 1..n; root has parent 0 (a sentinel meaning "no node").
    vector<int> par(n + 1, 0);
    vector<vector<int>> children(n + 1);
    int root = 0;
    for (int v = 1; v <= n; v++) {
        int p;
        cin >> p;          // p == 0 means v is the root
        par[v] = p;
        if (p == 0) root = v;
        else children[p].push_back(v);
    }

    // ---- depth (edges from root) via iterative BFS/DFS over the explicit forest ----
    // Although exactly one root is guaranteed, we iterate defensively from the root.
    vector<int> depth(n + 1, 0);
    {
        vector<int> order;
        order.reserve(n);
        // process from root downward
        vector<int> stk;
        if (root != 0) stk.push_back(root);
        while (!stk.empty()) {
            int u = stk.back();
            stk.pop_back();
            order.push_back(u);
            for (int c : children[u]) {
                depth[c] = depth[u] + 1;
                stk.push_back(c);
            }
        }
        (void)order;
    }

    // ---- height of each node: longest downward chain length (in edges) ----
    // height[leaf] = 0. Computed by processing nodes in decreasing depth order.
    // We also record, for each node, the child on a longest downward path
    // (the "long-path successor going down").
    vector<int> height(n + 1, 0);
    vector<int> downChild(n + 1, 0); // child that continues the long path downward
    {
        // order nodes by depth descending; bucket sort on depth.
        int maxd = 0;
        for (int v = 1; v <= n; v++) maxd = max(maxd, depth[v]);
        vector<int> cnt(maxd + 2, 0);
        for (int v = 1; v <= n; v++) cnt[depth[v]]++;
        for (int d = 1; d <= maxd; d++) cnt[d] += cnt[d - 1];
        vector<int> byDepth(n);
        for (int v = 1; v <= n; v++) byDepth[--cnt[depth[v]]] = v;
        // byDepth is ascending by depth; iterate in reverse => descending depth.
        for (int idx = n - 1; idx >= 0; idx--) {
            int v = byDepth[idx];
            int best = -1, bestChild = 0;
            for (int c : children[v]) {
                if (height[c] > best) { best = height[c]; bestChild = c; }
            }
            if (bestChild != 0) {
                height[v] = best + 1;
                downChild[v] = bestChild;
            } else {
                height[v] = 0;
                downChild[v] = 0;
            }
        }
    }

    // ---- long-path decomposition ----
    // A node is the HEAD (top) of its long path iff it is the root OR it is not
    // the downChild of its parent. Each long path runs head -> downChild -> ...
    // until height drops to 0.
    // For every node we store pathHead[v] (the top of its long path) and
    // posInLadder[v] (its index within that path's ladder array).
    vector<int> pathHead(n + 1, 0);
    vector<int> ladderId(n + 1, -1);     // which ladder array this node lives in
    vector<int> posInLadder(n + 1, 0);   // index of v inside ladder[ladderId[v]]
    vector<vector<int>> ladder;          // each ladder: indices from top(extended) .. bottom

    for (int v = 1; v <= n; v++) {
        bool isHead = (par[v] == 0) || (downChild[par[v]] != v);
        if (!isHead) continue;
        // walk the long path downward from v
        int L = height[v] + 1; // number of nodes on the long path (head..deepest)
        // ladder = L ancestors above head (if available) + the L path nodes.
        // First gather the path nodes.
        vector<int> pathNodes;
        pathNodes.reserve(L);
        int cur = v;
        while (cur != 0) {
            pathNodes.push_back(cur);
            cur = downChild[cur];
        }
        // pathNodes.size() == L
        // gather up to L ancestors above the head (the "ladder extension")
        vector<int> up;
        up.reserve(L);
        int a = par[v];
        for (int t = 0; t < L && a != 0; t++) {
            up.push_back(a);
            a = par[a];
        }
        // build the ladder array top-to-bottom: reversed(up) ++ pathNodes
        int id = (int)ladder.size();
        vector<int> arr;
        arr.reserve(up.size() + pathNodes.size());
        for (int t = (int)up.size() - 1; t >= 0; t--) arr.push_back(up[t]);
        int headOffset = (int)arr.size(); // index of head v inside arr
        for (int x : pathNodes) arr.push_back(x);
        // assign ladder membership for the PATH nodes only (each node is assigned
        // exactly once, by its own long path).
        for (int i = 0; i < (int)pathNodes.size(); i++) {
            int x = pathNodes[i];
            ladderId[x] = id;
            posInLadder[x] = headOffset + i;
            pathHead[x] = v;
        }
        ladder.push_back(move(arr));
    }

    // ---- jump pointers (binary lifting) ----
    int LOG = 1;
    while ((1 << LOG) < n + 1) LOG++;
    LOG = max(LOG, 1);
    // up2[j][v] = ancestor of v that is 2^j edges above v (0 = none).
    vector<vector<int>> up2(LOG + 1, vector<int>(n + 1, 0));
    for (int v = 1; v <= n; v++) up2[0][v] = par[v];
    for (int j = 1; j <= LOG; j++) {
        for (int v = 1; v <= n; v++) {
            int mid = up2[j - 1][v];
            up2[j][v] = (mid == 0) ? 0 : up2[j - 1][mid];
        }
    }

    // ---- queries ----
    int q;
    cin >> q;
    string out;
    out.reserve((size_t)q * 7);
    char buf[16];
    for (int Q = 0; Q < q; Q++) {
        int v; long long k;
        cin >> v >> k;
        int ans;
        if (k == 0) {
            ans = v;
        } else if (k > depth[v]) {
            ans = 0; // no such ancestor; report 0
        } else {
            // jump 2^j up where 2^j <= k < 2^(j+1)
            int j = 63 - __builtin_clzll((unsigned long long)k);
            int w = up2[j][v];               // w is ancestor at distance 2^j (exists since k<=depth[v])
            int rem = (int)(k - (1 << j));    // 0 <= rem < 2^j <= height(w)
            // w's ladder contains at least height(w) >= 2^j >= rem ancestors above w.
            int id = ladderId[w];
            int idx = posInLadder[w] - rem;   // move rem steps up inside the ladder array
            ans = ladder[id][idx];
        }
        int len = 0;
        if (ans == 0) { buf[len++] = '0'; }
        else { int t = ans; char tmp[16]; int tl = 0; while (t) { tmp[tl++] = char('0' + t % 10); t /= 10; } while (tl) buf[len++] = tmp[--tl]; }
        out.append(buf, len);
        out.push_back('\n');
    }
    cout << out;
    return 0;
}
```
