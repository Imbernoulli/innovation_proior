// TIER: strong
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, K;
vector<int> p, s, eu, ev, ew, eg;
vector<vector<int>> adj;

ll totalCost(const vector<int>& c) {
    ll F = 0;
    for (int i = 1; i <= n; i++) F += (ll)s[i] * abs(c[i] - p[i]);
    for (int i = 0; i < m; i++) {
        int diff = abs(c[eu[i]] - c[ev[i]]);
        int pen = eg[i] - diff; if (pen < 0) pen = 0;
        F += (ll)ew[i] * pen;
    }
    return F;
}

// best-response local search: move each beacon to its cost-minimizing channel
// (interference + retuning) until a full sweep makes no change.
void localSearch(vector<int>& c) {
    bool improved = true;
    int sweeps = 0;
    while (improved && sweeps++ < 60) {
        improved = false;
        for (int u = 1; u <= n; u++) {
            ll best = LLONG_MAX; int bc = c[u];
            for (int ch = 1; ch <= K; ch++) {
                ll cost = (ll)s[u] * abs(ch - p[u]);
                for (int id : adj[u]) {
                    int v = eu[id] == u ? ev[id] : eu[id];
                    int diff = abs(ch - c[v]);
                    int pen = eg[id] - diff; if (pen < 0) pen = 0;
                    cost += (ll)ew[id] * pen;
                }
                if (cost < best) { best = cost; bc = ch; }
            }
            if (bc != c[u]) { c[u] = bc; improved = true; }
        }
    }
}

int main() {
    if (scanf("%d %d %d", &n, &m, &K) != 3) return 0;
    p.resize(n + 1); s.resize(n + 1);
    for (int i = 1; i <= n; i++) scanf("%d %d", &p[i], &s[i]);
    eu.resize(m); ev.resize(m); ew.resize(m); eg.resize(m);
    adj.assign(n + 1, {});
    for (int i = 0; i < m; i++) {
        scanf("%d %d %d %d", &eu[i], &ev[i], &ew[i], &eg[i]);
        adj[eu[i]].push_back(i);
        adj[ev[i]].push_back(i);
    }

    // greedy warm start (same as greedy solution)
    vector<int> g0(n + 1, 0);
    for (int u = 1; u <= n; u++) {
        ll best = LLONG_MAX; int bc = 1;
        for (int ch = 1; ch <= K; ch++) {
            ll cost = (ll)s[u] * abs(ch - p[u]);
            for (int id : adj[u]) {
                int v = eu[id] == u ? ev[id] : eu[id];
                if (g0[v] == 0) continue;
                int diff = abs(ch - g0[v]);
                int pen = eg[id] - diff; if (pen < 0) pen = 0;
                cost += (ll)ew[id] * pen;
            }
            if (cost < best) { best = cost; bc = ch; }
        }
        g0[u] = bc;
    }

    mt19937 rng(2246789123u);
    vector<int> best; ll bestF = LLONG_MAX;
    int restarts = 10;
    for (int r = 0; r < restarts; r++) {
        vector<int> c(n + 1);
        if (r == 0) {
            c = g0;                                   // greedy warm start
        } else if (r == 1) {
            for (int u = 1; u <= n; u++) c[u] = p[u]; // home-channel start
        } else {
            for (int u = 1; u <= n; u++) c[u] = (int)(rng() % K) + 1;
        }
        localSearch(c);
        ll F = totalCost(c);
        if (F < bestF) { bestF = F; best = c; }
    }

    for (int i = 1; i <= n; i++) printf("%d%c", best[i], i < n ? ' ' : '\n');
    return 0;
}
