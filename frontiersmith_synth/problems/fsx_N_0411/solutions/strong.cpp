// TIER: strong
// Balanced breadth-first spanning tree grown from the hub (cheapest available
// child first, degree <= D). BFS level order makes the tree bushy / shallow
// (a near-complete D-ary tree) -> every rooted subtree is small -> LOW stress,
// while still using cheap edges. Balances cost against the robustness surrogate,
// so it beats the min-cost greedy on the composite objective.
#include <bits/stdc++.h>
using namespace std;

int par[6005];
int find(int x){ while(par[x]!=x){ par[x]=par[par[x]]; x=par[x]; } return x; }

int main() {
    int n, m, D; long long L;
    if (scanf("%d %d %d %lld", &n, &m, &D, &L) != 4) return 0;
    vector<int> eu(m + 1), ev(m + 1);
    vector<long long> ew(m + 1);
    for (int j = 1; j <= m; j++) scanf("%d %d %lld", &eu[j], &ev[j], &ew[j]);

    // adjacency: (cost, neighbor, edge_index), sorted by cost ascending
    vector<vector<array<long long,3>>> adj(n);
    for (int j = 1; j <= m; j++) {
        adj[eu[j]].push_back({ew[j], (long long)ev[j], (long long)j});
        adj[ev[j]].push_back({ew[j], (long long)eu[j], (long long)j});
    }
    for (int v = 0; v < n; v++) sort(adj[v].begin(), adj[v].end());

    vector<int> deg(n, 0);
    vector<char> inTree(n, 0);
    vector<int> ptr(n, 0);
    vector<int> chosen;

    // BFS balanced growth from hub
    queue<int> q;
    inTree[0] = 1; q.push(0);
    while (!q.empty()) {
        int u = q.front(); q.pop();
        while (deg[u] < D && ptr[u] < (int)adj[u].size()) {
            auto& e = adj[u][ptr[u]]; ptr[u]++;
            int x = (int)e[1];
            if (inTree[x]) continue;
            inTree[x] = 1; deg[u]++; deg[x]++;
            chosen.push_back((int)e[2]);
            q.push(x);
        }
    }

    // completion via DSU over chosen + backbone edges (safety)
    for (int i = 0; i < n; i++) par[i] = i;
    for (int e : chosen) par[find(eu[e])] = find(ev[e]);
    for (int i = 1; i <= n - 1; i++) {
        if (!inTree[i - 1] || !inTree[i]) continue;
        if (find(i - 1) != find(i) && deg[i - 1] < D && deg[i] < D) {
            par[find(i - 1)] = find(i);
            deg[i - 1]++; deg[i]++;
            chosen.push_back(i);
        }
    }
    // attach any still-detached node to any in-tree neighbor with spare degree
    for (int pass = 0; pass < 3; pass++) {
        for (int v = 0; v < n; v++) {
            if (inTree[v]) continue;
            for (auto& e : adj[v]) {
                int u = (int)e[1];
                if (inTree[u] && deg[u] < D) {
                    inTree[v] = 1; deg[u]++; deg[v]++;
                    chosen.push_back((int)e[2]);
                    par[find(u)] = find(v);
                    break;
                }
            }
        }
    }

    // verify spanning; if not, fall back to the always-feasible backbone
    bool full = true;
    for (int v = 0; v < n; v++) if (!inTree[v]) { full = false; break; }
    if (!full) {
        printf("%d\n", n - 1);
        for (int i = 1; i <= n - 1; i++) printf("%d\n", i);
        return 0;
    }

    printf("%d\n", (int)chosen.size());
    for (int e : chosen) printf("%d\n", e);
    return 0;
}
