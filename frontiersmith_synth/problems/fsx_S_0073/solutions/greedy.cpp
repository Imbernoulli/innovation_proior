// TIER: greedy
// One-pass proxy heuristic: repeatedly collapse the intermediate chamber on the
// CURRENT fastest route that has the largest total incident passage weight
// (a cheap structural proxy, NOT the true resulting detour), as long as it keeps
// the mouth and vault connected. No re-evaluation of the actual objective.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, s, t, k;
struct AdjE { int to; ll w; };
vector<vector<AdjE>> g;
vector<ll> incW; // total incident weight per chamber

// returns dist and fills `par` for path reconstruction; ignores dead chambers
ll dijkstra(const vector<char>& dead, vector<int>& par) {
    vector<ll> dist(n + 1, LLONG_MAX);
    par.assign(n + 1, -1);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<pair<ll,int>>> pq;
    dist[s] = 0; pq.push({0, s});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (auto& e : g[u]) {
            if (dead[e.to]) continue;
            ll nd = d + e.w;
            if (nd < dist[e.to]) { dist[e.to] = nd; par[e.to] = u; pq.push({nd, e.to}); }
        }
    }
    return dist[t];
}

int main() {
    if (scanf("%d %d %d %d %d", &n, &m, &s, &t, &k) != 5) return 0;
    g.assign(n + 1, {});
    incW.assign(n + 1, 0);
    for (int i = 0; i < m; i++) {
        int u, v; ll w; scanf("%d %d %lld", &u, &v, &w);
        g[u].push_back({v, w}); g[v].push_back({u, w});
        incW[u] += w; incW[v] += w;
    }

    vector<char> dead(n + 1, 0);
    vector<int> chosen;

    for (int step = 0; step < k; step++) {
        vector<int> par;
        ll d = dijkstra(dead, par);
        if (d == LLONG_MAX) break;
        // gather intermediate chambers on the current fastest route
        vector<int> path;
        int cur = t;
        while (cur != -1 && cur != s) { path.push_back(cur); cur = par[cur]; }
        // pick the collapsible on-route chamber with max incident weight
        int best = -1; ll bestW = -1;
        for (int c : path) {
            if (c == s || c == t || dead[c]) continue;
            if (incW[c] > bestW) { bestW = incW[c]; best = c; }
        }
        if (best == -1) break;
        // commit only if connectivity survives
        dead[best] = 1;
        vector<int> tmp;
        if (dijkstra(dead, tmp) == LLONG_MAX) { dead[best] = 0; break; }
        chosen.push_back(best);
    }

    printf("%d\n", (int)chosen.size());
    for (int c : chosen) printf("%d\n", c);
    return 0;
}
