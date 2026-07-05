// TIER: strong
// Local search with seeded random restarts. Restart 0 uses the sequential min-conflict
// greedy init; later restarts use random inits. Each restart repeatedly re-tunes every
// station to its locally best channel until no station moves (min-conflict descent).
// Keep the assignment with the lowest total interference cost across all restarts.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N, M, C;
vector<vector<pair<int,ll>>> adj;
vector<int> eu, ev;
vector<ll> ew;

ll total(const vector<int>& col) {
    ll F = 0;
    for (int i = 0; i < M; i++) {
        int d = abs(col[eu[i]] - col[ev[i]]);
        if (d == 0) F += ew[i];
        else if (d == 1) F += ew[i] / 2;
    }
    return F;
}

ll localCost(int u, int c, const vector<int>& col) {
    ll cost = 0;
    for (auto& pr : adj[u]) {
        int d = abs(c - col[pr.first]);
        if (d == 0) cost += pr.second;
        else if (d == 1) cost += pr.second / 2;
    }
    return cost;
}

mt19937 rng(0xC0FFEEu);

int main() {
    if (scanf("%d %d %d", &N, &M, &C) != 3) return 0;
    adj.assign(N + 1, {});
    eu.resize(M); ev.resize(M); ew.resize(M);
    for (int i = 0; i < M; i++) {
        scanf("%d %d %lld", &eu[i], &ev[i], &ew[i]);
        adj[eu[i]].push_back({ev[i], ew[i]});
        adj[ev[i]].push_back({eu[i], ew[i]});
    }

    vector<int> best;
    ll bestF = LLONG_MAX;
    int restarts = 8;

    for (int rs = 0; rs < restarts; rs++) {
        vector<int> col(N + 1, 1);
        if (rs == 0) {
            // sequential min-conflict greedy init
            vector<int> assigned(N + 1, 0);
            for (int u = 1; u <= N; u++) {
                ll b = LLONG_MAX; int bc = 1;
                for (int c = 1; c <= C; c++) {
                    ll cost = 0;
                    for (auto& pr : adj[u]) {
                        int v = pr.first;
                        if (!assigned[v]) continue;
                        int d = abs(c - col[v]);
                        if (d == 0) cost += pr.second;
                        else if (d == 1) cost += pr.second / 2;
                    }
                    if (cost < b) { b = cost; bc = c; }
                }
                col[u] = bc; assigned[u] = 1;
            }
        } else {
            for (int u = 1; u <= N; u++) col[u] = 1 + (int)(rng() % C);
        }

        // min-conflict descent
        bool improved = true; int iter = 0;
        while (improved && iter < 80) {
            improved = false; iter++;
            for (int u = 1; u <= N; u++) {
                int cur = col[u];
                ll bcost = localCost(u, cur, col);
                int bc = cur;
                for (int c = 1; c <= C; c++) {
                    if (c == cur) continue;
                    ll cost = localCost(u, c, col);
                    if (cost < bcost) { bcost = cost; bc = c; }
                }
                if (bc != cur) { col[u] = bc; improved = true; }
            }
        }

        ll F = total(col);
        if (F < bestF) { bestF = F; best = col; }
    }

    for (int u = 1; u <= N; u++) printf("%d\n", best[u]);
    return 0;
}
