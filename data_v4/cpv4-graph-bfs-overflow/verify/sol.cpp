#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<long long> w(n + 1);
    for (int v = 1; v <= n; v++) cin >> w[v];

    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int a, b;
        cin >> a >> b;
        adj[a].push_back(b);
        adj[b].push_back(a);
    }

    // BFS from city 1 over the unweighted graph to get hop-distances.
    const int INF = -1;
    vector<int> dist(n + 1, INF);
    queue<int> q;
    dist[1] = 0;
    q.push(1);
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (int v : adj[u]) {
            if (dist[v] == INF) {
                dist[v] = dist[u] + 1;
                q.push(v);
            }
        }
    }

    // Sum dist[v] * w[v] over all reachable cities. With dist up to n-1
    // (~2e5) and w up to 1e6, a single product reaches ~2e11 and the whole
    // sum reaches ~2e16, so 64-bit accumulation is mandatory; int overflows.
    long long total = 0;
    for (int v = 1; v <= n; v++) {
        if (dist[v] != INF) {
            total += (long long)dist[v] * w[v];
        }
    }

    cout << total << "\n";
    return 0;
}
