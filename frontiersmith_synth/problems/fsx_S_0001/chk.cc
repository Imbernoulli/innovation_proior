#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, s, t, k;
vector<vector<pair<int,ll>>> adj; // node -> (to, weight) with parallel edge index
vector<int> eidx_of;              // (unused placeholder)

// adjacency stores edge index so we can skip removed edges
struct AdjE { int to; ll w; int idx; };
vector<vector<AdjE>> g;

ll dijkstra(const vector<char>& removed) {
    vector<ll> dist(n + 1, LLONG_MAX);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<pair<ll,int>>> pq;
    dist[s] = 0; pq.push({0, s});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        if (u == t) return d;
        for (auto& e : g[u]) {
            if (removed[e.idx]) continue;
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
        g[u].push_back({v, w, i});
        g[v].push_back({u, w, i});
    }

    // internal baseline: original shortest supply route (no links closed)
    ll B = dijkstra(vector<char>(m + 1, 0));
    if (B == LLONG_MAX || B <= 0) quitf(_fail, "bad instance: B=%lld", B);

    // ---- read & validate participant's closed-link set ----
    int r = ouf.readInt(0, k, "r");
    vector<char> removed(m + 1, 0);
    for (int i = 0; i < r; i++) {
        int idx = ouf.readInt(1, m, "edgeIndex");
        if (removed[idx]) quitf(_wa, "link %d closed more than once", idx);
        removed[idx] = 1;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    ll F = dijkstra(removed);
    if (F == LLONG_MAX) quitf(_wa, "closing these links disconnects mill %d from bakery %d", s, t);

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
