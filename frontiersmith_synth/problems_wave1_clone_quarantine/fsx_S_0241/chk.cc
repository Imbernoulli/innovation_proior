#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, s, t, k;
struct AdjE { int to; ll w; };
vector<vector<AdjE>> g;   // node -> (to, weight)

// shortest s-t distance in the subgraph induced by NON-removed nodes.
// removedNode[v]=1 means station v is powered down (skipped entirely).
ll dijkstra(const vector<char>& removedNode) {
    vector<ll> dist(n + 1, LLONG_MAX);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<pair<ll,int>>> pq;
    dist[s] = 0; pq.push({0, s});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        if (u == t) return d;
        for (auto& e : g[u]) {
            if (removedNode[e.to]) continue;
            ll nd = d + e.w;
            if (nd < dist[e.to]) { dist[e.to] = nd; pq.push({nd, e.to}); }
        }
    }
    return dist[t];
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    n = inf.readInt();
    m = inf.readInt();
    s = inf.readInt();
    t = inf.readInt();
    k = inf.readInt();

    g.assign(n + 1, {});
    for (int i = 1; i <= m; i++) {
        int u = inf.readInt(), v = inf.readInt();
        ll w = inf.readInt();
        g[u].push_back({v, w});
        g[v].push_back({u, w});
    }

    // internal baseline: original fastest relay route (no stations powered down)
    vector<char> none(n + 1, 0);
    ll B = dijkstra(none);
    if (B == LLONG_MAX || B <= 0) quitf(_fail, "bad instance: B=%lld", B);

    // ---- read & validate participant's powered-down station set ----
    int r = ouf.readInt(0, k, "r");
    vector<char> removedNode(n + 1, 0);
    for (int i = 0; i < r; i++) {
        int idx = ouf.readInt(1, n, "stationIndex");
        if (idx == s || idx == t)
            quitf(_wa, "cannot power down inlet %d or sink %d (got %d)", s, t, idx);
        if (removedNode[idx])
            quitf(_wa, "station %d powered down more than once", idx);
        removedNode[idx] = 1;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    ll F = dijkstra(removedNode);
    if (F == LLONG_MAX)
        quitf(_wa, "powering these stations down disconnects inlet %d from sink %d", s, t);

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
