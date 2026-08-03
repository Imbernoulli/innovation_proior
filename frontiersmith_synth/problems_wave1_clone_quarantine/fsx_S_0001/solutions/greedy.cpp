// TIER: greedy
// One-pass heuristic: repeatedly close the MOST EXPENSIVE link lying on the
// current shortest supply route, provided the mill stays connected to the bakery.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, s, t, k;
struct AdjE { int to; ll w; int idx; };
vector<vector<AdjE>> g;

// returns shortest dist to t; fills parentEdge for path reconstruction
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
    if (scanf("%d %d %d %d %d", &n, &m, &s, &t, &k) != 5) return 0;
    g.assign(n + 1, {});
    vector<ll> ew(m + 1);
    for (int i = 1; i <= m; i++) {
        int u, v; ll w; scanf("%d %d %lld", &u, &v, &w);
        g[u].push_back({v, w, i}); g[v].push_back({u, w, i}); ew[i] = w;
    }
    vector<char> removed(m + 1, 0);
    vector<int> chosen;
    for (int it = 0; it < k; it++) {
        vector<int> pe, pn;
        ll d = dijkstra(removed, pe, pn);
        if (d == LLONG_MAX) break;
        // collect edges on shortest path
        vector<int> pathEdges;
        int cur = t;
        while (cur != s && pe[cur] != -1) { pathEdges.push_back(pe[cur]); cur = pn[cur]; }
        // sort path edges by weight desc, close the first that keeps connectivity
        sort(pathEdges.begin(), pathEdges.end(),
             [&](int a, int b){ return ew[a] > ew[b]; });
        bool done = false;
        for (int e : pathEdges) {
            removed[e] = 1;
            if (connected(removed)) { chosen.push_back(e); done = true; break; }
            removed[e] = 0;
        }
        if (!done) break;
    }
    printf("%d\n", (int)chosen.size());
    for (int e : chosen) printf("%d\n", e);
    return 0;
}
