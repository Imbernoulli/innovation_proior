#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    if (!(cin >> n >> m)) return 0;
    vector<long long> c(n + 1);
    for (int v = 1; v <= n; v++) cin >> c[v];

    vector<vector<pair<int,long long>>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v; long long w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
    }

    const long long INF = LLONG_MAX / 4;
    vector<long long> dist(n + 1, INF);

    // You start at station 1 at time 0.
    // Being present at station v at any time t with t >= c[v] is forbidden,
    // so an arrival time t at v is LEGAL only if t < c[v] (strict).
    // Departure happens at the same instant as arrival (no waiting helps),
    // so the only constraint per node is the strict-inequality arrival check.

    // Start station must itself be legal at time 0.
    if (0 < c[1]) {
        dist[1] = 0;
    }

    priority_queue<pair<long long,int>, vector<pair<long long,int>>,
                   greater<pair<long long,int>>> pq;
    if (dist[1] == 0) pq.push({0, 1});

    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d != dist[u]) continue;
        for (auto [v, w] : adj[u]) {
            long long nd = d + w;
            // Arriving at v at time nd is allowed only if nd < c[v] (strict boundary).
            if (nd < c[v] && nd < dist[v]) {
                dist[v] = nd;
                pq.push({nd, v});
            }
        }
    }

    if (dist[n] >= INF) cout << -1 << "\n";
    else cout << dist[n] << "\n";
    return 0;
}
