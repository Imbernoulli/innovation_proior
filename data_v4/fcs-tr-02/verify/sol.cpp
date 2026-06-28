#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long L;
    if (!(cin >> n >> L)) return 0;

    // Adjacency: for each vertex, list of (neighbor, weight).
    vector<vector<pair<int,int>>> adj(n + 1);
    for (int e = 0; e < n - 1; e++) {
        int u, v, w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }

    // Frequency array indexed by distance value in [0, L].
    // Edge weights are positive, so only distances <= L can ever combine to
    // exactly L; we never index outside [0, L].
    vector<long long> freq(L + 1, 0);
    freq[0] = 1; // the centroid itself sits at distance 0 (counts centroid-endpoint paths)

    vector<char> removed(n + 1, 0); // vertices already used as a centroid
    vector<int> sub(n + 1, 0);       // subtree sizes (recomputed per component)
    vector<int> par(n + 1, 0);       // parent in the current rooted traversal

    long long answer = 0;

    // Compute subtree sizes of the component containing `root`, return component size.
    // Fills sub[] and par[] for vertices in this component (rooted at root).
    auto computeSizes = [&](int root) -> int {
        vector<int> ord;
        ord.reserve(64);
        vector<pair<int,int>> stk;
        stk.push_back({root, 0});
        while (!stk.empty()) {
            auto [u, p] = stk.back(); stk.pop_back();
            par[u] = p;
            ord.push_back(u);
            for (auto [v, w] : adj[u]) {
                if (v != p && !removed[v]) stk.push_back({v, u});
            }
        }
        for (int u : ord) sub[u] = 1;
        for (int i = (int)ord.size() - 1; i >= 0; i--) {
            int u = ord[i], p = par[u];
            if (p != 0) sub[p] += sub[u];
        }
        return (int)ord.size();
    };

    // Find the centroid of a component, given precomputed sub[]/par[] rooted at `root`.
    auto findCentroid = [&](int root, int total) -> int {
        int cur = root, prev = 0;
        while (true) {
            int nxt = -1;
            for (auto [v, w] : adj[cur]) {
                if (v == prev || removed[v]) continue;
                // size of the component "behind" v when we cut edge (cur,v):
                // if v is a child of cur in the root-tree, that side is sub[v];
                // otherwise (v == par[cur]) that side is total - sub[cur].
                int sz = (par[v] == cur) ? sub[v] : (total - sub[cur]);
                if (sz > total / 2) { nxt = v; break; }
            }
            if (nxt == -1) break;
            prev = cur;
            cur = nxt;
        }
        return cur;
    };

    // Gather distances of all nodes in the branch rooted at `start` (parent `parent0`),
    // starting from baseDist; keep only distances <= L.
    vector<long long> dists;
    auto gatherDists = [&](int start, int parent0, long long baseDist) {
        dists.clear();
        vector<tuple<int,int,long long>> stk;
        stk.push_back({start, parent0, baseDist});
        while (!stk.empty()) {
            auto [u, p, d] = stk.back(); stk.pop_back();
            dists.push_back(d); // d <= L guaranteed by the push guard below / base check
            for (auto [v, w] : adj[u]) {
                if (v == p || removed[v]) continue;
                long long nd = d + w;
                if (nd <= L) stk.push_back({v, u, nd});
            }
        }
    };

    // Iterative centroid decomposition over an explicit stack of component entry points.
    vector<int> compStack;
    compStack.push_back(1);

    while (!compStack.empty()) {
        int entry = compStack.back(); compStack.pop_back();
        if (removed[entry]) continue;
        int total = computeSizes(entry);
        int c = findCentroid(entry, total);

        // Count paths through centroid c. freq holds only freq[0] = 1 right now.
        // For each branch of c: first ADD over already-inserted branches (and the
        // centroid via freq[0]), then insert this branch's distances. This counts
        // every cross-branch pair exactly once and centroid-endpoint paths once.
        vector<long long> inserted;
        for (auto [child, w] : adj[c]) {
            if (removed[child]) continue;
            long long base = (long long)w;
            if (base > L) continue; // whole branch starts beyond L; no node can match
            gatherDists(child, c, base);
            for (long long d : dists) {
                long long need = L - d;
                if (need >= 0) answer += freq[need]; // need <= L since d >= 0
            }
            for (long long d : dists) {
                freq[d] += 1;
                inserted.push_back(d);
            }
        }
        for (long long d : inserted) freq[d] -= 1; // restore freq to {freq[0]=1}

        // Remove the centroid and recurse into the pieces around it.
        removed[c] = 1;
        for (auto [v, w] : adj[c]) {
            if (!removed[v]) compStack.push_back(v);
        }
    }

    cout << answer << "\n";
    return 0;
}
