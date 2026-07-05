// TIER: strong
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, K;
vector<int> eu, ev, ew, ed;
vector<vector<int>> adj;

ll totalCost(const vector<int>& c) {
    ll F = 0;
    for (int i = 0; i < m; i++) {
        int diff = abs(c[eu[i]] - c[ev[i]]);
        int pen = ed[i] - diff; if (pen < 0) pen = 0;
        F += (ll)ew[i] * pen;
    }
    return F;
}

// best-response local search: move each person to its cost-minimizing cohort
// until a full sweep makes no change.
ll localSearch(vector<int>& c) {
    bool improved = true;
    while (improved) {
        improved = false;
        for (int u = 1; u <= n; u++) {
            ll best = LLONG_MAX; int bc = c[u];
            for (int ch = 1; ch <= K; ch++) {
                ll cost = 0;
                for (int id : adj[u]) {
                    int v = eu[id] == u ? ev[id] : eu[id];
                    int diff = abs(ch - c[v]);
                    int pen = ed[id] - diff; if (pen < 0) pen = 0;
                    cost += (ll)ew[id] * pen;
                }
                if (cost < best) { best = cost; bc = ch; }
            }
            if (bc != c[u]) { c[u] = bc; improved = true; }
        }
    }
    return totalCost(c);
}

int main() {
    if (scanf("%d %d %d", &n, &m, &K) != 3) return 0;
    eu.resize(m); ev.resize(m); ew.resize(m); ed.resize(m);
    adj.assign(n + 1, {});
    for (int i = 0; i < m; i++) {
        scanf("%d %d %d %d", &eu[i], &ev[i], &ew[i], &ed[i]);
        adj[eu[i]].push_back(i);
        adj[ev[i]].push_back(i);
    }

    mt19937 rng(987654321u);
    vector<int> best; ll bestF = LLONG_MAX;
    int restarts = 12;
    for (int r = 0; r < restarts; r++) {
        vector<int> c(n + 1, 0);
        if (r == 0) {
            // greedy warm start
            for (int u = 1; u <= n; u++) {
                ll bb = LLONG_MAX; int bc = 1;
                for (int ch = 1; ch <= K; ch++) {
                    ll cost = 0;
                    for (int id : adj[u]) {
                        int v = eu[id] == u ? ev[id] : eu[id];
                        if (c[v] == 0) continue;
                        int diff = abs(ch - c[v]);
                        int pen = ed[id] - diff; if (pen < 0) pen = 0;
                        cost += (ll)ew[id] * pen;
                    }
                    if (cost < bb) { bb = cost; bc = ch; }
                }
                c[u] = bc;
            }
        } else {
            for (int u = 1; u <= n; u++) c[u] = (int)(rng() % K) + 1;
        }
        ll F = localSearch(c);
        if (F < bestF) { bestF = F; best = c; }
    }

    for (int i = 1; i <= n; i++) printf("%d%c", best[i], i < n ? ' ' : '\n');
    return 0;
}
