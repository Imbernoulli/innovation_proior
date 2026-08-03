// TIER: strong
// Greedy init + min-conflicts local search with a few seeded random restarts.
// Each restart runs sweeps that move every rig to its least-penalty channel until
// no improvement; keeps the best assignment across restarts.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static int n, m, C;
static vector<vector<array<int,3>>> adj; // (nbr, p, q)

// penalty contributed by rig u given current colours (for its incident edges)
static inline ll localPenalty(int u, int ch, const vector<int>& col) {
    ll add = 0;
    for (auto& e : adj[u]) {
        int nc = col[e[0]];
        int diff = abs(nc - ch);
        if (diff == 0) add += e[1];
        else if (diff == 1) add += e[2];
    }
    return add;
}

// run min-conflicts sweeps starting from `col`; returns total interference.
static ll refine(vector<int>& col) {
    bool improved = true;
    int sweeps = 0;
    while (improved && sweeps < 40) {
        improved = false; sweeps++;
        for (int u = 1; u <= n; u++) {
            ll cur = localPenalty(u, col[u], col);
            ll best = cur; int bestc = col[u];
            for (int ch = 1; ch <= C; ch++) {
                if (ch == col[u]) continue;
                ll v = localPenalty(u, ch, col);
                if (v < best) { best = v; bestc = ch; }
            }
            if (bestc != col[u]) { col[u] = bestc; improved = true; }
        }
    }
    // total interference (each edge once)
    ll F = 0;
    for (int u = 1; u <= n; u++)
        for (auto& e : adj[u])
            if (e[0] > u) {
                int diff = abs(col[u] - col[e[0]]);
                if (diff == 0) F += e[1];
                else if (diff == 1) F += e[2];
            }
    return F;
}

int main() {
    if (scanf("%d %d %d", &n, &m, &C) != 3) return 0;
    adj.assign(n + 1, {});
    vector<ll> wsum(n + 1, 0);
    for (int i = 0; i < m; i++) {
        int u, v, p, q; scanf("%d %d %d %d", &u, &v, &p, &q);
        adj[u].push_back({v, p, q});
        adj[v].push_back({u, p, q});
        wsum[u] += p; wsum[v] += p;
    }

    // ---- greedy init (weight-ordered) ----
    vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int a, int b){ return wsum[a] > wsum[b]; });

    vector<int> gcol(n + 1, 0);
    for (int idx = 0; idx < n; idx++) {
        int u = order[idx];
        ll best = LLONG_MAX; int bestc = 1;
        for (int ch = 1; ch <= C; ch++) {
            ll add = 0;
            for (auto& e : adj[u]) {
                int nc = gcol[e[0]];
                if (nc == 0) continue;
                int diff = abs(nc - ch);
                if (diff == 0) add += e[1];
                else if (diff == 1) add += e[2];
            }
            if (add < best) { best = add; bestc = ch; }
        }
        gcol[u] = bestc;
    }

    vector<int> best = gcol;
    ll bestF = refine(best);

    // ---- seeded random restarts ----
    std::mt19937 rng(12345u);
    int restarts = 6;
    for (int r = 0; r < restarts; r++) {
        vector<int> col(n + 1);
        for (int u = 1; u <= n; u++) col[u] = (int)(rng() % C) + 1;
        ll F = refine(col);
        if (F < bestF) { bestF = F; best = col; }
    }

    for (int i = 1; i <= n; i++) printf("%d%c", best[i], i == n ? '\n' : ' ');
    return 0;
}
