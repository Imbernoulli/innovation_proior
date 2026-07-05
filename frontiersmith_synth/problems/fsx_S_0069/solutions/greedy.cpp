// TIER: greedy
// Saturation-order greedy: process radios by interference degree, assign each the
// channel that minimizes added interference against already-assigned neighbors.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, K, M;
    if (scanf("%d %d %d", &N, &K, &M) != 3) return 0;
    struct E { int to, s, p; };
    vector<vector<E>> adj(N + 1);
    vector<int> deg(N + 1, 0);
    for (int i = 0; i < M; i++) {
        int u, v, s, p;
        scanf("%d %d %d %d", &u, &v, &s, &p);
        adj[u].push_back({v, s, p});
        adj[v].push_back({u, s, p});
        deg[u]++; deg[v]++;
    }

    vector<int> order(N);
    for (int i = 0; i < N; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (deg[a] != deg[b]) return deg[a] > deg[b];
        return a < b;
    });

    vector<int> c(N + 1, 0); // 0 = unassigned
    for (int idx = 0; idx < N; idx++) {
        int u = order[idx];
        long long best = LLONG_MAX;
        int bestCh = 1;
        for (int ch = 1; ch <= K; ch++) {
            long long cost = 0;
            for (auto& e : adj[u]) {
                if (c[e.to] == 0) continue;
                int diff = abs(ch - c[e.to]);
                int sh = e.s - diff;
                if (sh > 0) cost += (long long)e.p * sh;
            }
            if (cost < best) { best = cost; bestCh = ch; }
        }
        c[u] = bestCh;
    }

    for (int i = 1; i <= N; i++) printf("%d%c", c[i], i == N ? '\n' : ' ');
    return 0;
}
