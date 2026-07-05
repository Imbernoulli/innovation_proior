// TIER: greedy
// One-pass greedy: process stations in index order; assign each the channel that adds the
// least separation cost against its already-assigned neighbors. No revisiting.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M, K;
    if (scanf("%d %d %d", &N, &M, &K) != 3) return 0;

    vector<vector<array<int,3>>> adj(N + 1); // for node -> (nbr, g, w)
    for (int i = 0; i < M; i++) {
        int u, v, g, w;
        if (scanf("%d %d %d %d", &u, &v, &g, &w) != 4) break;
        adj[u].push_back({v, g, w});
        adj[v].push_back({u, g, w});
    }

    vector<int> col(N + 1, 0); // 0 = unassigned
    for (int j = 1; j <= N; j++) {
        long long best = LLONG_MAX;
        int bestc = 1;
        for (int c = 1; c <= K; c++) {
            long long cost = 0;
            for (auto &e : adj[j]) {
                int nb = e[0];
                if (col[nb] == 0) continue;
                int d = abs(c - col[nb]);
                int deficit = e[1] - d;
                if (deficit > 0) cost += (long long)e[2] * deficit;
            }
            if (cost < best) { best = cost; bestc = c; }
        }
        col[j] = bestc;
    }

    for (int j = 1; j <= N; j++) printf("%d\n", col[j]);
    return 0;
}
