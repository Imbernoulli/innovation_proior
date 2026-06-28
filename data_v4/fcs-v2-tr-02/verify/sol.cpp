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
