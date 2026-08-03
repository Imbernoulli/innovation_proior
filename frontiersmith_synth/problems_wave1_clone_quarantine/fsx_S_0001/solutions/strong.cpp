// TIER: strong
// Best-improvement interdiction with randomized multi-restart local search.
// Each restart: k rounds, each round evaluate CLOSING every link on the current
// shortest route by the actual increase it causes (re-solve Dijkstra), then commit
// the best (with occasional random exploration). Keep the best link set over all restarts.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, s, t, k;
struct AdjE { int to; ll w; int idx; };
vector<vector<AdjE>> g;

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

ll spDist(const vector<char>& removed) {
    vector<int> pe, pn; return dijkstra(removed, pe, pn);
}

mt19937 rng(987654321u);

int main() {
    if (scanf("%d %d %d %d %d", &n, &m, &s, &t, &k) != 5) return 0;
    g.assign(n + 1, {});
    for (int i = 1; i <= m; i++) {
        int u, v; ll w; scanf("%d %d %lld", &u, &v, &w);
        g[u].push_back({v, w, i}); g[v].push_back({u, w, i});
    }

    vector<int> bestSet;
    ll bestF = -1;

    int restarts = 12;
    for (int rs = 0; rs < restarts; rs++) {
        vector<char> removed(m + 1, 0);
        vector<int> chosen;
        for (int it = 0; it < k; it++) {
            vector<int> pe, pn;
            ll d = dijkstra(removed, pe, pn);
            if (d == LLONG_MAX) break;
            // path edges of current shortest route
            vector<int> pathEdges;
            int cur = t;
            while (cur != s && pe[cur] != -1) { pathEdges.push_back(pe[cur]); cur = pn[cur]; }
            if (pathEdges.empty()) break;

            // evaluate closing each path edge by the ACTUAL resulting shortest dist
            ll bestGain = -1; int bestEdge = -1;
            vector<pair<ll,int>> cand; // (resulting dist, edge)
            for (int e : pathEdges) {
                removed[e] = 1;
                ll nd = spDist(removed);
                removed[e] = 0;
                if (nd == LLONG_MAX) continue; // would disconnect
                cand.push_back({nd, e});
                if (nd > bestGain) { bestGain = nd; bestEdge = e; }
            }
            if (bestEdge == -1) break; // every path edge disconnects -> stop

            int pick = bestEdge;
            // occasional exploration: pick a random feasible candidate instead of the best
            if (rs > 0 && !cand.empty() && (rng() % 100) < 30) {
                pick = cand[rng() % cand.size()].second;
            }
            removed[pick] = 1; chosen.push_back(pick);
        }
        ll F = spDist(removed);
        if (F != LLONG_MAX && F > bestF) { bestF = F; bestSet = chosen; }
    }

    printf("%d\n", (int)bestSet.size());
    for (int e : bestSet) printf("%d\n", e);
    return 0;
}
