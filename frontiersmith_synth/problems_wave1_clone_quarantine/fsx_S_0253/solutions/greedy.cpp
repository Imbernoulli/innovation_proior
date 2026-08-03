// TIER: greedy
// Cost-benefit greedy: repeatedly look at the current shortest den->watering route
// and clear the affordable corridor on it with the best length-per-clearing-cost
// ratio, provided the den stays connected to the watering ground. One pass, no
// re-evaluation of the true resulting route.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, s, t; ll C;
struct AdjE { int to; ll w; int idx; };
vector<vector<AdjE>> g;
vector<ll> ew, ec;

ll dijkstra(const vector<char>& removed, vector<int>& parEdge, vector<int>& parNode) {
    vector<ll> dist(n + 1, LLONG_MAX);
    parEdge.assign(n + 1, -1); parNode.assign(n + 1, -1);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<pair<ll,int>>> pq;
    dist[s] = 0; pq.push({0, s});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (auto& e : g[u]) {
            if (removed[e.idx]) continue;
            ll nd = d + e.w;
            if (nd < dist[e.to]) {
                dist[e.to] = nd; parEdge[e.to] = e.idx; parNode[e.to] = u;
                pq.push({nd, e.to});
            }
        }
    }
    return dist[t];
}

bool connected(const vector<char>& removed) {
    vector<int> pe, pn;
    return dijkstra(removed, pe, pn) != LLONG_MAX;
}

int main() {
    if (scanf("%d %d %d %d %lld", &n, &m, &s, &t, &C) != 5) return 0;
    g.assign(n + 1, {});
    ew.assign(m + 1, 0); ec.assign(m + 1, 0);
    for (int i = 1; i <= m; i++) {
        int u, v; ll w, c; scanf("%d %d %lld %lld", &u, &v, &w, &c);
        g[u].push_back({v, w, i}); g[v].push_back({u, w, i});
        ew[i] = w; ec[i] = c;
    }
    vector<char> removed(m + 1, 0);
    vector<int> chosen;
    ll budget = C;
    while (true) {
        vector<int> pe, pn;
        ll d = dijkstra(removed, pe, pn);
        if (d == LLONG_MAX) break;
        vector<int> pathEdges;
        int cur = t;
        while (cur != s && pe[cur] != -1) { pathEdges.push_back(pe[cur]); cur = pn[cur]; }
        // pick affordable path edge with best length/cost ratio that keeps connectivity
        int best = -1; double bestRatio = -1.0;
        for (int e : pathEdges) {
            if (ec[e] > budget) continue;
            double ratio = (double)ew[e] / (double)ec[e];
            if (ratio > bestRatio) {
                removed[e] = 1;
                bool ok = connected(removed);
                removed[e] = 0;
                if (ok) { bestRatio = ratio; best = e; }
            }
        }
        if (best == -1) break;
        removed[best] = 1; budget -= ec[best]; chosen.push_back(best);
    }
    printf("%d\n", (int)chosen.size());
    for (int e : chosen) printf("%d\n", e);
    return 0;
}
