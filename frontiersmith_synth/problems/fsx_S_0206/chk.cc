#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, D, r;
vector<ll> cost;
struct AdjE { int to; ll w; };
vector<vector<AdjE>> g;
vector<int> shops;

// multi-source Dijkstra from the set of built depots; returns dist to every node
// (dist to nearest depot). Caps at r+1 conceptually but we just run full Dijkstra.
vector<ll> multiDijkstra(const vector<int>& depots) {
    vector<ll> dist(n + 1, LLONG_MAX);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<pair<ll,int>>> pq;
    for (int u : depots) { if (dist[u] > 0) { dist[u] = 0; pq.push({0, u}); } }
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        if (d > r) continue;              // no need to expand beyond radius
        for (auto& e : g[u]) {
            ll nd = d + e.w;
            if (nd < dist[e.to]) { dist[e.to] = nd; pq.push({nd, e.to}); }
        }
    }
    return dist;
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    n = inf.readInt();
    m = inf.readInt();
    D = inf.readInt();
    r = inf.readInt();

    cost.assign(n + 1, 0);
    for (int u = 1; u <= n; u++) cost[u] = inf.readInt();

    g.assign(n + 1, {});
    for (int i = 0; i < m; i++) {
        int u = inf.readInt(), v = inf.readInt();
        ll w = inf.readInt();
        g[u].push_back({v, w});
        g[v].push_back({u, w});
    }

    shops.resize(D);
    for (int i = 0; i < D; i++) shops[i] = inf.readInt();

    // internal baseline B: build a depot in every shop district (always feasible,
    // each shop is at distance 0 from its own depot).
    ll B = 0;
    for (int d : shops) B += cost[d];
    if (B <= 0) quitf(_fail, "bad instance: B=%lld", B);

    // ---- read & validate participant's depot set ----
    int q = ouf.readInt(0, n, "q");
    vector<char> built(n + 1, 0);
    vector<int> depots;
    ll F = 0;
    for (int i = 0; i < q; i++) {
        int u = ouf.readInt(1, n, "depot");
        if (built[u]) quitf(_wa, "district %d built more than once", u);
        built[u] = 1;
        depots.push_back(u);
        F += cost[u];
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // coverage check: every shop must have a depot within travel time r
    vector<ll> dist = multiDijkstra(depots);
    for (int d : shops) {
        if (dist[d] > r) quitf(_wa, "shop %d is not supplied within radius r=%d (nearest depot dist %lld)",
                                d, r, dist[d] == LLONG_MAX ? -1 : dist[d]);
    }

    if (F <= 0) quitf(_wa, "no feasible non-empty covering (F=%lld)", F);

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
