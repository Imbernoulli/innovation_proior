#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, s, t, k;
struct AdjE { int to; ll w; };
vector<vector<AdjE>> g;

// shortest s-t path ignoring any chamber marked in `dead`
ll dijkstra(const vector<char>& dead) {
    vector<ll> dist(n + 1, LLONG_MAX);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<pair<ll,int>>> pq;
    dist[s] = 0; pq.push({0, s});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        if (u == t) return d;
        for (auto& e : g[u]) {
            if (dead[e.to]) continue;
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

    // internal baseline: original fastest route (no chambers collapsed)
    vector<char> none(n + 1, 0);
    ll B = dijkstra(none);
    if (B == LLONG_MAX || B <= 0) quitf(_fail, "bad instance: B=%lld", B);

    // ---- read & validate participant's collapsed-chamber set ----
    int r = ouf.readInt(0, k, "r");
    vector<char> dead(n + 1, 0);
    for (int i = 0; i < r; i++) {
        int idx = ouf.readInt(1, n, "chamberIndex");
        if (idx == s) quitf(_wa, "cannot collapse the cave mouth s=%d", s);
        if (idx == t) quitf(_wa, "cannot collapse the crystal vault t=%d", t);
        if (dead[idx]) quitf(_wa, "chamber %d collapsed more than once", idx);
        dead[idx] = 1;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    ll F = dijkstra(dead);
    if (F == LLONG_MAX) quitf(_wa, "collapsing these chambers cuts vault %d off from mouth %d", t, s);

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
