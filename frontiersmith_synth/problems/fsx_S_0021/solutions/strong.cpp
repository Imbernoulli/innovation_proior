// TIER: strong
// Greedy init + deterministic min-conflicts local search with a few perturbed
// restarts; keeps the best assignment found. Beats plain greedy on most tests.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, K;
vector<vector<array<int,3>>> adj; // (neighbor, p, q)

// penalty of assigning channel ch to node u given current assignment c
static inline ll nodeCost(int u, int ch, const vector<int>& c) {
    ll cost = 0;
    for (auto& e : adj[u]) {
        int w = e[0]; int cw = c[w];
        if (cw == ch) cost += e[1];
        else if (abs(cw - ch) == 1) cost += e[2];
    }
    return cost;
}

ll totalCost(const vector<int>& c) {
    ll t = 0;
    for (int u = 1; u <= n; u++) t += nodeCost(u, c[u], c);
    return t / 2; // each pair counted twice
}

int main() {
    scanf("%d %d %d", &n, &m, &K);
    adj.assign(n + 1, {});
    for (int i = 0; i < m; i++) {
        int u, v, p, q; scanf("%d %d %d %d", &u, &v, &p, &q);
        adj[u].push_back({v, p, q});
        adj[v].push_back({u, p, q});
    }

    // greedy init in index order
    vector<int> c(n + 1, 1);
    for (int u = 1; u <= n; u++) {
        ll best = LLONG_MAX; int bc = 1;
        for (int ch = 1; ch <= K; ch++) {
            ll cost = 0;
            for (auto& e : adj[u]) {
                int w = e[0]; if (w > u) continue; // only placed nodes (1..u-1)
                if (c[w] == ch) cost += e[1];
                else if (abs(c[w] - ch) == 1) cost += e[2];
            }
            if (cost < best) { best = cost; bc = ch; }
        }
        c[u] = bc;
    }

    auto localSearch = [&](vector<int>& cur) {
        for (int sweep = 0; sweep < 2000; sweep++) {
            bool improved = false;
            for (int u = 1; u <= n; u++) {
                ll cbest = nodeCost(u, cur[u], cur); int chbest = cur[u];
                for (int ch = 1; ch <= K; ch++) {
                    if (ch == cur[u]) continue;
                    ll cc = nodeCost(u, ch, cur);
                    if (cc < cbest) { cbest = cc; chbest = ch; }
                }
                if (chbest != cur[u]) { cur[u] = chbest; improved = true; }
            }
            if (!improved) break;
        }
    };

    localSearch(c);
    vector<int> best = c;
    ll bestF = totalCost(best);

    // deterministic perturbed restarts (seeded LCG, no wall clock)
    unsigned long long seed = 0x9E3779B97F4A7C15ull ^ ((unsigned long long)n << 20) ^ (unsigned long long)m;
    auto nextRand = [&]() { seed = seed * 6364136223846793005ull + 1442695040888963407ull; return (unsigned)(seed >> 33); };

    for (int r = 0; r < 12 && bestF > 0; r++) {
        vector<int> cur = best;
        int kicks = 1 + (n / 5);
        for (int t = 0; t < kicks; t++) {
            int u = 1 + nextRand() % n;
            cur[u] = 1 + nextRand() % K;
        }
        localSearch(cur);
        ll f = totalCost(cur);
        if (f < bestF) { bestF = f; best = cur; }
    }

    for (int i = 1; i <= n; i++) printf("%d%c", best[i], i < n ? ' ' : '\n');
    return 0;
}
