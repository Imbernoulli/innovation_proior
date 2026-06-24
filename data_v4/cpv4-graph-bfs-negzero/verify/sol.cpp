#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, s;
    if (!(cin >> n >> m >> s)) return 0;     // n=0 is impossible (see constraints), but guard anyway

    vector<long long> w(n + 1);
    for (int i = 1; i <= n; i++) cin >> w[i];

    vector<vector<int>> adj(n + 1);
    for (int e = 0; e < m; e++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // BFS from s: dist[v] = shortest hop count, -1 if unreachable.
    vector<int> dist(n + 1, -1);
    queue<int> q;
    dist[s] = 0;
    q.push(s);
    int maxDist = 0;
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (int v : adj[u]) {
            if (dist[v] == -1) {
                dist[v] = dist[u] + 1;
                maxDist = max(maxDist, dist[v]);
                q.push(v);
            }
        }
    }

    // layerSum[d] = sum of brightness of reachable nodes at exact distance d.
    // Only reachable nodes (dist != -1) contribute; unreachable nodes are ignored.
    vector<long long> layerSum(maxDist + 1, 0);
    for (int v = 1; v <= n; v++) {
        if (dist[v] != -1) layerSum[dist[v]] += w[v];
    }

    // Answer: maximum layer-sum over all NON-EMPTY layers. Layer 0 (the source) is
    // always non-empty, so at least one layer exists; the answer may be negative.
    long long best = LLONG_MIN;
    for (int d = 0; d <= maxDist; d++) {
        // Every layer 0..maxDist is non-empty here (BFS produces no gaps), but guard anyway.
        // layerSum[d] could be 0 even for a non-empty layer (a single zero-brightness node),
        // so we cannot use 0 to detect emptiness.
        best = max(best, layerSum[d]);
    }

    cout << best << "\n";
    return 0;
}
