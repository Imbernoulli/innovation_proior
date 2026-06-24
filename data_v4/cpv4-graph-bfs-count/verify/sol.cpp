#include <bits/stdc++.h>
using namespace std;

static const long long MOD = 1000000007LL;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, s, t;
    if (!(cin >> n >> m >> s >> t)) return 0;

    vector<vector<int>> adj(n + 1);
    for (int e = 0; e < m; e++) {
        int u, v;
        cin >> u >> v;
        if (u == v) continue;              // ignore self-loops: they never help a shortest path
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    const int INF = INT_MAX;
    vector<int> dist(n + 1, INF);
    vector<long long> ways(n + 1, 0);

    // BFS from s; layer-by-layer so every vertex is finalized in nondecreasing distance order.
    queue<int> q;
    dist[s] = 0;
    ways[s] = 1;
    q.push(s);
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (int w : adj[u]) {
            if (dist[w] == INF) {              // first time we reach w
                dist[w] = dist[u] + 1;
                ways[w] = ways[u];             // start its count from this predecessor
                q.push(w);
            } else if (dist[w] == dist[u] + 1) {
                // another shortest predecessor on the previous layer
                ways[w] = (ways[w] + ways[u]) % MOD;
            }
            // dist[w] == dist[u] (same layer) or dist[w] < dist[u]: contributes nothing.
        }
    }

    long long ans = (dist[t] == INF) ? 0 : ways[t] % MOD;
    cout << ans << "\n";
    return 0;
}
