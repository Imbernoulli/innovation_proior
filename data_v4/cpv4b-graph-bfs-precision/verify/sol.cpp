#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    if (!(cin >> n >> m)) return 0;

    // adjacency: for each undirected edge, store neighbour and the gain factor p/q.
    vector<vector<array<long long,3>>> adj(n); // {to, p, q}
    for (int i = 0; i < m; i++) {
        int u, v; long long p, q;
        cin >> u >> v >> p >> q;
        adj[u].push_back({(long long)v, p, q});
        adj[v].push_back({(long long)u, p, q});
    }

    const int INF = INT_MAX;
    vector<int> dist(n, INF);
    // BFS from node 0 to get min-hop distance (the routing rule: fewest hops only).
    queue<int> bfs;
    dist[0] = 0; bfs.push(0);
    vector<int> order; order.reserve(n);
    while (!bfs.empty()) {
        int u = bfs.front(); bfs.pop();
        order.push_back(u);
        for (auto &e : adj[u]) {
            int v = (int)e[0];
            if (dist[v] == INF) { dist[v] = dist[u] + 1; bfs.push(v); }
        }
    }

    if (dist[n-1] == INF) { cout << -1 << "\n"; return 0; }

    // Among all min-hop paths, maximize product of gains. Store the best product at
    // each node as an UNREDUCED fraction (num, den) in long long. Distance < n <= 20,
    // each factor <= 9, so num,den <= 9^19 < 2^63. Comparisons cross-multiply, which
    // can reach 9^38 ~ 1.8e36 -> needs __int128.
    vector<long long> bnum(n, 0), bden(n, 1);
    vector<char> has(n, 0);
    bnum[0] = 1; bden[0] = 1; has[0] = 1; // empty path: gain 1

    // process nodes in BFS order = non-decreasing distance, so predecessors at
    // distance d-1 are finalized before nodes at distance d are relaxed.
    for (int u : order) {
        if (!has[u]) continue;
        for (auto &e : adj[u]) {
            int v = (int)e[0];
            if (dist[v] != dist[u] + 1) continue; // only forward edges on a min-hop path
            long long cn = bnum[u] * e[1]; // candidate numerator
            long long cd = bden[u] * e[2]; // candidate denominator
            if (!has[v]) { bnum[v] = cn; bden[v] = cd; has[v] = 1; continue; }
            // compare cn/cd  >  bnum[v]/bden[v]  <=>  cn*bden[v] > bnum[v]*cd
            __int128 lhs = (__int128)cn * bden[v];
            __int128 rhs = (__int128)bnum[v] * cd;
            if (lhs > rhs) { bnum[v] = cn; bden[v] = cd; }
        }
    }

    long long P = bnum[n-1], Q = bden[n-1];
    long long g = std::__gcd(P, Q);
    P /= g; Q /= g;
    cout << P << "/" << Q << "\n";
    return 0;
}
