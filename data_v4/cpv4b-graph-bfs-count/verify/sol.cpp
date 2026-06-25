#include <bits/stdc++.h>
using namespace std;

const long long MOD = 1000000007LL;

int main() {
    int n, m;
    if (!(cin >> n >> m)) return 0;
    vector<vector<int>> adj(n + 1);
    vector<pair<int,int>> edges(m);
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        edges[i] = {u, v};
    }
    // Build simple-graph adjacency: collapse parallel edges, drop self-loops,
    // because a "route" is a station sequence, so multiplicity does not matter.
    {
        vector<set<int>> tmp(n + 1);
        for (auto &e : edges) {
            int u = e.first, v = e.second;
            if (u == v) continue;            // self-loop: never on a shortest simple step
            tmp[u].insert(v);
            tmp[v].insert(u);
        }
        for (int u = 1; u <= n; u++)
            adj[u].assign(tmp[u].begin(), tmp[u].end());
    }

    const int INF = INT_MAX;
    vector<int> dist(n + 1, INF);
    vector<long long> ways(n + 1, 0);
    dist[1] = 0;
    ways[1] = 1 % MOD;
    queue<int> q;
    q.push(1);
    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int v : adj[u]) {
            if (dist[v] == INF) {
                dist[v] = dist[u] + 1;
                ways[v] = ways[u];
                q.push(v);
            } else if (dist[v] == dist[u] + 1) {
                // v already discovered on the next layer: u is one valid predecessor.
                ways[v] = (ways[v] + ways[u]) % MOD;
            }
            // dist[v] == dist[u] (same layer) or dist[v] < dist[u]: NOT a predecessor.
        }
    }

    if (dist[n] == INF) {
        cout << 0 << "\n";
    } else {
        cout << ways[n] % MOD << "\n";
    }
    return 0;
}
