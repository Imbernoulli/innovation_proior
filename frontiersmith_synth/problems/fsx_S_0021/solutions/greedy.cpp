// TIER: greedy
// Sequential greedy coloring: process collectors 1..n; each takes the channel
// minimizing added penalty against already-assigned neighbors only.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m, K;
    scanf("%d %d %d", &n, &m, &K);
    vector<vector<array<int,3>>> adj(n + 1); // (neighbor, p, q)
    for (int i = 0; i < m; i++) {
        int u, v, p, q; scanf("%d %d %d %d", &u, &v, &p, &q);
        adj[u].push_back({v, p, q});
        adj[v].push_back({u, p, q});
    }
    vector<int> c(n + 1, 0);
    for (int u = 1; u <= n; u++) {
        long long best = LLONG_MAX; int bestc = 1;
        for (int ch = 1; ch <= K; ch++) {
            long long cost = 0;
            for (auto& e : adj[u]) {
                int w = e[0]; if (c[w] == 0) continue;
                if (c[w] == ch) cost += e[1];
                else if (abs(c[w] - ch) == 1) cost += e[2];
            }
            if (cost < best) { best = cost; bestc = ch; }
        }
        c[u] = bestc;
    }
    for (int i = 1; i <= n; i++) printf("%d%c", c[i], i < n ? ' ' : '\n');
    return 0;
}
