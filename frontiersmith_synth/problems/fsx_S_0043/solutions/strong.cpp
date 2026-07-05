// TIER: strong
// Degree-constrained Kruskal, then cap-aware cycle-swap local search on the spanning tree.
#include <bits/stdc++.h>
using namespace std;

struct DSU {
    vector<int> p;
    DSU(int n) : p(n + 1) { for (int i = 0; i <= n; i++) p[i] = i; }
    int find(int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
    bool uni(int a, int b) { a = find(a); b = find(b); if (a == b) return false; p[a] = b; return true; }
};

int N, M;
vector<int> cap, deg;
// tree adjacency: for each node, list of (neighbor, edgeCost)
vector<vector<pair<int,int>>> adj;

int main() {
    if (scanf("%d %d", &N, &M) != 2) return 0;
    cap.assign(N + 1, 0);
    for (int i = 1; i <= N; i++) scanf("%d", &cap[i]);
    vector<array<int,3>> E(M); // w,u,v
    for (int k = 0; k < M; k++) {
        int u, v, w; scanf("%d %d %d", &u, &v, &w);
        E[k] = {w, u, v};
    }
    sort(E.begin(), E.end());

    // Full-graph adjacency (cost, neighbor), sorted -- used by the NN fallback builder.
    vector<vector<pair<int,int>>> gadj(N + 1);
    for (auto &e : E) { gadj[e[1]].push_back({e[0], e[2]}); gadj[e[2]].push_back({e[0], e[1]}); }
    for (int i = 1; i <= N; i++) sort(gadj[i].begin(), gadj[i].end());

    deg.assign(N + 1, 0);
    DSU dsu(N);
    adj.assign(N + 1, {});
    int comps = N;
    bool spanning = false;

    auto addTree = [&](int u, int v, int w) {
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
        deg[u]++; deg[v]++;
    };

    // Pass 1: degree-constrained Kruskal.
    for (auto &e : E) {
        int w = e[0], u = e[1], v = e[2];
        if (dsu.find(u) == dsu.find(v)) continue;
        if (deg[u] >= cap[u] || deg[v] >= cap[v]) continue;
        dsu.uni(u, v); addTree(u, v, w); comps--;
        if (comps == 1) { spanning = true; break; }
    }
    // Pass 2: repair.
    if (!spanning) {
        for (auto &e : E) {
            int w = e[0], u = e[1], v = e[2];
            if (dsu.find(u) == dsu.find(v)) continue;
            if (deg[u] >= cap[u] || deg[v] >= cap[v]) continue;
            dsu.uni(u, v); addTree(u, v, w); comps--;
            if (comps == 1) { spanning = true; break; }
        }
    }

    // If Kruskal painted itself into a corner, rebuild a guaranteed-spanning tree via a
    // nearest-neighbour construction, then let local search polish it below.
    if (!spanning) {
        adj.assign(N + 1, {});
        deg.assign(N + 1, 0);
        vector<char> vis(N + 1, 0);
        vector<int> covered = {1};
        vis[1] = 1; int cnt = 1, cur = 1;
        while (cnt < N) {
            int best = -1, bestw = 0;
            if (deg[cur] < cap[cur])
                for (auto &pr : gadj[cur]) {
                    if (!vis[pr.second] && deg[pr.second] < cap[pr.second]) { best = pr.second; bestw = pr.first; break; }
                }
            if (best != -1) {
                addTree(cur, best, bestw); vis[best] = 1; covered.push_back(best); cnt++; cur = best; continue;
            }
            int bu = -1, bw = -1, bc = INT_MAX;
            for (int u : covered) {
                if (deg[u] >= cap[u]) continue;
                for (auto &pr : gadj[u]) {
                    if (vis[pr.second] || deg[pr.second] >= cap[pr.second]) continue;
                    if (pr.first < bc) { bc = pr.first; bu = u; bw = pr.second; }
                    break;
                }
            }
            if (bu != -1) { addTree(bu, bw, bc); vis[bw] = 1; covered.push_back(bw); cnt++; cur = bw; continue; }
            break;
        }
        if (cnt != N) {
            // last-resort feasible output: the guaranteed chain
            printf("%d\n", N - 1);
            for (int i = 1; i <= N - 1; i++) printf("%d %d\n", i, i + 1);
            return 0;
        }
    }

    // ---- local search: cycle-swap improvement respecting caps ----
    // For a non-tree edge (u,v,c): find the path u..v in the tree, take its max-cost edge
    // (mp,mq,mw). If mw > c and dropping (mp,mq) then adding (u,v) keeps all caps, swap.
    // Repeat in passes until no improvement.
    auto findPathMax = [&](int s, int t, int &mp, int &mq, int &mw) -> bool {
        // BFS from s to t recording parent + edge cost, then walk back for max edge
        vector<int> par(N + 1, 0), pw(N + 1, 0);
        vector<char> vis(N + 1, 0);
        queue<int> q; q.push(s); vis[s] = 1; par[s] = -1;
        while (!q.empty()) {
            int x = q.front(); q.pop();
            if (x == t) break;
            for (auto &pr : adj[x]) {
                int y = pr.first;
                if (!vis[y]) { vis[y] = 1; par[y] = x; pw[y] = pr.second; q.push(y); }
            }
        }
        if (!vis[t]) return false;
        mw = -1;
        int cur = t;
        while (par[cur] != -1) {
            int pr = par[cur];
            if (pw[cur] > mw) { mw = pw[cur]; mp = pr; mq = cur; }
            cur = pr;
        }
        return mw >= 0;
    };

    auto removeEdge = [&](int a, int b) {
        for (size_t i = 0; i < adj[a].size(); i++)
            if (adj[a][i].first == b) { adj[a].erase(adj[a].begin() + i); break; }
        for (size_t i = 0; i < adj[b].size(); i++)
            if (adj[b][i].first == a) { adj[b].erase(adj[b].begin() + i); break; }
        deg[a]--; deg[b]--;
    };

    int passes = 0, maxPasses = 60;
    bool improved = true;
    while (improved && passes < maxPasses) {
        improved = false;
        passes++;
        for (auto &e : E) {
            int c = e[0], u = e[1], v = e[2];
            // skip if already a tree edge
            bool isTree = false;
            for (auto &pr : adj[u]) if (pr.first == v) { isTree = true; break; }
            if (isTree) continue;
            int mp = 0, mq = 0, mw = -1;
            if (!findPathMax(u, v, mp, mq, mw)) continue;
            if (mw <= c) continue; // no gain
            // simulate degree after dropping (mp,mq): both -1, then adding (u,v): +1 each
            int du = deg[u] - ((mp == u) || (mq == u) ? 1 : 0);
            int dv = deg[v] - ((mp == v) || (mq == v) ? 1 : 0);
            if (du + 1 > cap[u]) continue;
            if (dv + 1 > cap[v]) continue;
            // perform swap
            removeEdge(mp, mq);
            addTree(u, v, c);
            improved = true;
        }
    }

    // emit final tree edges (each once)
    vector<pair<int,int>> out;
    for (int u = 1; u <= N; u++)
        for (auto &pr : adj[u])
            if (u < pr.first) out.push_back({u, pr.first});

    printf("%d\n", (int)out.size());
    for (auto &pr : out) printf("%d %d\n", pr.first, pr.second);
    return 0;
}
