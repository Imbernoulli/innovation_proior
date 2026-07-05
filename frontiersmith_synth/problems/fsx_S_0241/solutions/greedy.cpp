// TIER: greedy
// One-pass heuristic: repeatedly power down the HIGHEST-DEGREE interior station lying on
// the current fastest relay route, provided the inlet stays connected to the sink.
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

int main() {
    if (scanf("%d %d %d %d %d", &n, &m, &s, &t, &k) != 5) return 0;
    g.assign(n + 1, {});
    for (int i = 1; i <= m; i++) {
        int u, v; ll w; scanf("%d %d %lld", &u, &v, &w);
        g[u].push_back({v, w}); g[v].push_back({u, w});
    }
    vector<int> deg(n + 1, 0);
    for (int u = 1; u <= n; u++) deg[u] = (int)g[u].size();

    vector<char> rem(n + 1, 0);
    vector<int> chosen;
    for (int it = 0; it < k; it++) {
        vector<int> par;
        ll d = dijkstra(rem, par);
        if (d == LLONG_MAX) break;
        // collect interior nodes on the fastest route (exclude s and t)
        vector<int> pathNodes;
        int cur = t;
        while (cur != s && par[cur] != -1) {
            if (cur != t) pathNodes.push_back(cur);
            cur = par[cur];
        }
        // sort interior stations by degree desc; power down the first keeping connectivity
        sort(pathNodes.begin(), pathNodes.end(),
             [&](int a, int b){ return deg[a] > deg[b]; });
        bool done = false;
        for (int node : pathNodes) {
            rem[node] = 1;
            vector<int> tmp;
            if (dijkstra(rem, tmp) != LLONG_MAX) { chosen.push_back(node); done = true; break; }
            rem[node] = 0;
        }
        if (!done) break;
    }
    printf("%d\n", (int)chosen.size());
    for (int x : chosen) printf("%d\n", x);
    return 0;
}
