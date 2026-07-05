// TIER: greedy
// Sequential min-conflict: process stations 1..N in order; assign each the channel
// that minimizes added co-channel (w) + adjacent-channel (w/2) crosstalk against
// its already-assigned neighbors. One pass, no revisiting.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main() {
    int N, M, C;
    if (scanf("%d %d %d", &N, &M, &C) != 3) return 0;
    vector<vector<pair<int,ll>>> adj(N + 1);
    for (int i = 0; i < M; i++) {
        int u, v; ll w; scanf("%d %d %lld", &u, &v, &w);
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    vector<int> col(N + 1, 0); // 0 = unassigned
    for (int u = 1; u <= N; u++) {
        ll best = LLONG_MAX; int bc = 1;
        for (int c = 1; c <= C; c++) {
            ll cost = 0;
            for (auto& pr : adj[u]) {
                int v = pr.first;
                if (col[v] == 0) continue;
                int d = abs(c - col[v]);
                if (d == 0) cost += pr.second;
                else if (d == 1) cost += pr.second / 2;
            }
            if (cost < best) { best = cost; bc = c; }
        }
        col[u] = bc;
    }
    for (int u = 1; u <= N; u++) printf("%d\n", col[u]);
    return 0;
}
