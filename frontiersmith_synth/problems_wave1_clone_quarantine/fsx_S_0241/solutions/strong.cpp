// TIER: strong
// Best-improvement interdiction with randomized multi-restart local search.
// Each restart: k rounds; each round tentatively powers down every candidate station
// (interior nodes on the current fastest route AND their neighbours), re-solves the
// shortest route, and commits the station that maximizes latency (with occasional random
// exploration). Keep the best station set found over all restarts.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, s, t, k;
struct AdjE { int to; ll w; };
vector<vector<AdjE>> g;

ll dijkstra(const vector<char>& rem, vector<int>& par) {
    vector<ll> dist(n + 1, LLONG_MAX);
    par.assign(n + 1, -1);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<pair<ll,int>>> pq;
    dist[s] = 0; pq.push({0, s});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (auto& e : g[u]) {
            if (rem[e.to]) continue;
            ll nd = d + e.w;
            if (nd < dist[e.to]) { dist[e.to] = nd; par[e.to] = u; pq.push({nd, e.to}); }
        }
    }
    return dist[t];
}

ll spDist(const vector<char>& rem) { vector<int> p; return dijkstra(rem, p); }

mt19937 rng(1234567u);

int main() {
    if (scanf("%d %d %d %d %d", &n, &m, &s, &t, &k) != 5) return 0;
    g.assign(n + 1, {});
    for (int i = 1; i <= m; i++) {
        int u, v; ll w; scanf("%d %d %lld", &u, &v, &w);
        g[u].push_back({v, w}); g[v].push_back({u, w});
    }

    vector<int> bestSet;
    ll bestF = -1;

    int restarts = 14;
    for (int rs = 0; rs < restarts; rs++) {
        vector<char> rem(n + 1, 0);
        vector<int> chosen;
        for (int it = 0; it < k; it++) {
            vector<int> par;
            ll d = dijkstra(rem, par);
            if (d == LLONG_MAX) break;
            // candidate stations: interior route nodes + their neighbours (dedup)
            vector<int> pathNodes;
            int cur = t;
            while (cur != s && par[cur] != -1) {
                if (cur != t) pathNodes.push_back(cur);
                cur = par[cur];
            }
            if (pathNodes.empty()) break;
            vector<char> seen(n + 1, 0);
            vector<int> cands;
            for (int node : pathNodes) {
                if (!seen[node] && node != s && node != t && !rem[node]) { seen[node]=1; cands.push_back(node); }
                for (auto& e : g[node]) {
                    int v = e.to;
                    if (!seen[v] && v != s && v != t && !rem[v]) { seen[v]=1; cands.push_back(v); }
                }
            }
            // evaluate powering down each candidate by the ACTUAL resulting latency
            ll bestVal = -1; int bestNode = -1;
            vector<pair<ll,int>> feas;
            for (int node : cands) {
                rem[node] = 1;
                ll nd = spDist(rem);
                rem[node] = 0;
                if (nd == LLONG_MAX) continue; // would disconnect
                feas.push_back({nd, node});
                if (nd > bestVal) { bestVal = nd; bestNode = node; }
            }
            if (bestNode == -1) break; // every candidate disconnects -> stop
            int pick = bestNode;
            // occasional exploration on later restarts
            if (rs > 0 && !feas.empty() && (rng() % 100) < 30)
                pick = feas[rng() % feas.size()].second;
            rem[pick] = 1; chosen.push_back(pick);
        }
        ll F = spDist(rem);
        if (F != LLONG_MAX && F > bestF) { bestF = F; bestSet = chosen; }
    }

    printf("%d\n", (int)bestSet.size());
    for (int x : bestSet) printf("%d\n", x);
    return 0;
}
