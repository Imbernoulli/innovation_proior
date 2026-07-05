// TIER: strong
// Best-improvement local search with seeded multi-restart. At each step it
// evaluates the TRUE post-collapse shortest route for every collapsible chamber
// on the current fastest route and commits the genuine best (skipping any
// collapse that would disconnect mouth from vault). Several randomized restarts
// perturb the first choice to escape local optima; the best collapse set wins.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, s, t, k;
struct AdjE { int to; ll w; };
vector<vector<AdjE>> g;

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

// evaluate distance given a dead set (no path needed)
ll distOnly(const vector<char>& dead) {
    vector<int> par;
    return dijkstra(dead, par);
}

int main() {
    if (scanf("%d %d %d %d %d", &n, &m, &s, &t, &k) != 5) return 0;
    g.assign(n + 1, {});
    for (int i = 0; i < m; i++) {
        int u, v; ll w; scanf("%d %d %lld", &u, &v, &w);
        g[u].push_back({v, w}); g[v].push_back({u, w});
    }

    mt19937 rng(0xC0FFEEu ^ (unsigned)(n * 131 + m * 7 + k));

    vector<int> globalBest;
    ll globalScore = -1;

    int restarts = 6;
    for (int rs = 0; rs < restarts; rs++) {
        vector<char> dead(n + 1, 0);
        vector<int> chosen;

        for (int step = 0; step < k; step++) {
            vector<int> par;
            ll d = dijkstra(dead, par);
            if (d == LLONG_MAX) break;
            vector<int> path;
            int cur = t;
            while (cur != -1 && cur != s) { path.push_back(cur); cur = par[cur]; }

            // candidate collapsible chambers on the current route
            vector<int> cand;
            for (int c : path)
                if (c != s && c != t && !dead[c]) cand.push_back(c);
            if (cand.empty()) break;

            int pick = -1; ll pickVal = -1;
            if (rs > 0 && step == 0) {
                // perturbation: random valid first collapse that keeps connectivity
                shuffle(cand.begin(), cand.end(), rng);
                for (int c : cand) {
                    dead[c] = 1;
                    ll nd = distOnly(dead);
                    dead[c] = 0;
                    if (nd != LLONG_MAX) { pick = c; pickVal = nd; break; }
                }
            } else {
                // best-improvement: true resulting distance for each candidate
                for (int c : cand) {
                    dead[c] = 1;
                    ll nd = distOnly(dead);
                    dead[c] = 0;
                    if (nd == LLONG_MAX) continue; // would disconnect
                    if (nd > pickVal) { pickVal = nd; pick = c; }
                }
            }
            if (pick == -1) break;
            dead[pick] = 1;
            chosen.push_back(pick);
        }

        ll sc = distOnly(dead);
        if (sc == LLONG_MAX) continue;
        if (sc > globalScore) { globalScore = sc; globalBest = chosen; }
    }

    printf("%d\n", (int)globalBest.size());
    for (int c : globalBest) printf("%d\n", c);
    return 0;
}
