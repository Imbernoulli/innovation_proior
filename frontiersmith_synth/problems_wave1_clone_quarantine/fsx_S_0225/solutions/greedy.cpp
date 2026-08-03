// TIER: greedy
// One-pass greedy: process hydrophones in index order; assign each the channel that
// minimizes the added interference with its already-assigned neighbors.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m, K;
    scanf("%d %d %d", &n, &m, &K);
    vector<vector<array<int,3>>> adj(n + 1); // (nbr, w, d)
    for (int i = 0; i < m; i++) {
        int u, v, w, d;
        scanf("%d %d %d %d", &u, &v, &w, &d);
        adj[u].push_back({v, w, d});
        adj[v].push_back({u, w, d});
    }
    vector<int> c(n + 1, 0); // 0 = unassigned
    for (int i = 1; i <= n; i++) {
        long long best = LLONG_MAX; int bestCh = 1;
        for (int ch = 1; ch <= K; ch++) {
            long long cost = 0;
            for (auto &e : adj[i]) {
                int nb = e[0];
                if (c[nb] == 0) continue;
                int diff = abs(ch - c[nb]);
                int pen = e[2] - diff;
                if (pen > 0) cost += (long long)e[1] * pen;
            }
            if (cost < best) { best = cost; bestCh = ch; }
        }
        c[i] = bestCh;
    }
    for (int i = 1; i <= n; i++) printf("%d%c", c[i], i == n ? '\n' : ' ');
    return 0;
}
