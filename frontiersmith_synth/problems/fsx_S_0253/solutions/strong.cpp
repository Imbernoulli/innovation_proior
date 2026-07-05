// TIER: strong
// Best-improvement local search: at each step, for every affordable corridor on the
// current shortest den->watering route, tentatively clear it, recompute the TRUE
// resulting shortest route, and commit the clearing that maximizes it (tie-break on
// lower clearing cost so budget is spent efficiently). Keeps connectivity. Repeats
// until no affordable corridor improves the route. This actually evaluates the
// downstream detour instead of trusting a length/cost proxy, so it diverges from greedy.
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

ll dist_only(const vector<char>& removed) {
    vector<int> pe, pn;
    return dijkstra(removed, pe, pn);
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
        int best = -1; ll bestF = d; ll bestCost = LLONG_MAX;
        for (int e : pathEdges) {
            if (ec[e] > budget) continue;
            removed[e] = 1;
            ll f = dist_only(removed);
            removed[e] = 0;
            if (f == LLONG_MAX) continue; // would disconnect
            if (f > bestF || (f == bestF && ec[e] < bestCost)) {
                bestF = f; best = e; bestCost = ec[e];
            }
        }
        if (best == -1) break;
        removed[best] = 1; budget -= ec[best]; chosen.push_back(best);
    }
    printf("%d\n", (int)chosen.size());
    for (int e : chosen) printf("%d\n", e);
    return 0;
}
