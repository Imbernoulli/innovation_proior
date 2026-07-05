// TIER: greedy
// One-pass greedy coloring: process depots 1..n in index order; assign each depot the
// circuit that minimizes the added interference with already-colored neighbors.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m, C;
    if (scanf("%d %d %d", &n, &m, &C) != 3) return 0;
    vector<vector<pair<int,int>>> adj(n + 1); // (neighbor, weight)
    for (int i = 0; i < m; i++) {
        int u, v, w;
        scanf("%d %d %d", &u, &v, &w);
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    vector<int> col(n + 1, 0); // 0 = uncolored
    for (int i = 1; i <= n; i++) {
        vector<long long> cost(C + 1, 0);
        for (auto& e : adj[i]) {
            int nb = e.first;
            if (col[nb] != 0) cost[col[nb]] += e.second;
        }
        int best = 1;
        for (int c = 2; c <= C; c++)
            if (cost[c] < cost[best]) best = c;
        col[i] = best;
    }
    for (int i = 1; i <= n; i++) printf("%d ", col[i]);
    printf("\n");
    return 0;
}
