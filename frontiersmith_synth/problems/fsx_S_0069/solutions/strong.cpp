// TIER: strong
// Greedy warm-start + deterministic min-conflicts local search sweeps: repeatedly move
// each radio to the channel minimizing its local interference until no sweep improves.
#include <bits/stdc++.h>
using namespace std;

int N, K, M;
struct E { int to, s, p; };
vector<vector<E>> adj;
vector<int> c;

static inline long long localCost(int u, int ch) {
    long long cost = 0;
    for (auto& e : adj[u]) {
        int cv = c[e.to];
        if (cv == 0) continue;
        int diff = abs(ch - cv);
        int sh = e.s - diff;
        if (sh > 0) cost += (long long)e.p * sh;
    }
    return cost;
}

int main() {
    if (scanf("%d %d %d", &N, &K, &M) != 3) return 0;
    adj.assign(N + 1, {});
    vector<int> deg(N + 1, 0);
    for (int i = 0; i < M; i++) {
        int u, v, s, p;
        scanf("%d %d %d %d", &u, &v, &s, &p);
        adj[u].push_back({v, s, p});
        adj[v].push_back({u, s, p});
        deg[u]++; deg[v]++;
    }

    // ---- greedy warm start (saturation order) ----
    vector<int> order(N);
    for (int i = 0; i < N; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (deg[a] != deg[b]) return deg[a] > deg[b];
        return a < b;
    });
    c.assign(N + 1, 0);
    for (int idx = 0; idx < N; idx++) {
        int u = order[idx];
        long long best = LLONG_MAX; int bestCh = 1;
        for (int ch = 1; ch <= K; ch++) {
            long long cost = localCost(u, ch);
            if (cost < best) { best = cost; bestCh = ch; }
        }
        c[u] = bestCh;
    }

    // ---- min-conflicts local search: sweep in fixed order until convergence ----
    int maxSweeps = 60;
    for (int sweep = 0; sweep < maxSweeps; sweep++) {
        bool improved = false;
        for (int u = 1; u <= N; u++) {
            long long curCost = localCost(u, c[u]);
            long long best = curCost; int bestCh = c[u];
            for (int ch = 1; ch <= K; ch++) {
                if (ch == c[u]) continue;
                long long cost = localCost(u, ch);
                if (cost < best) { best = cost; bestCh = ch; }
            }
            if (bestCh != c[u]) { c[u] = bestCh; improved = true; }
        }
        if (!improved) break;
    }

    for (int i = 1; i <= N; i++) printf("%d%c", c[i], i == N ? '\n' : ' ');
    return 0;
}
