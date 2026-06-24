#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, s;
    if (!(cin >> n >> m >> s)) return 0;
    const long long MOD = 1000000007LL;

    vector<vector<pair<int,int>>> adj(n); // adj[u] = list of (to, weight)
    for (int i = 0; i < m; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
    }

    const long long INF = LLONG_MAX;
    vector<long long> dist(n, INF);
    vector<long long> cnt(n, 0); // number of shortest routes mod MOD

    priority_queue<pair<long long,int>, vector<pair<long long,int>>, greater<>> pq;
    dist[s] = 0;
    cnt[s] = 1;
    pq.push({0, s});

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (d > dist[u]) continue;             // stale entry: a shorter dist[u] is already final
        for (auto [v, w] : adj[u]) {
            long long nd = d + w;
            if (nd < dist[v]) {                // strictly shorter route to v: reset its count
                dist[v] = nd;
                cnt[v] = cnt[u];
                pq.push({nd, v});
            } else if (nd == dist[v]) {         // another shortest route to v: accumulate once
                cnt[v] = (cnt[v] + cnt[u]) % MOD;
            }
        }
    }

    for (int i = 0; i < n; i++) {
        if (dist[i] == INF) cout << -1 << "\n";
        else cout << (cnt[i] % MOD) << "\n";
    }
    return 0;
}
