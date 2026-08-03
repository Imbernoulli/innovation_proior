#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, s, t;
ll C; // clearing budget

struct AdjE { int to; ll w; int idx; };
vector<vector<AdjE>> g;
vector<ll> ecost; // clearing cost per edge index

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
    C = inf.readLong();

    g.assign(n + 1, {});
    ecost.assign(m + 1, 0);
    for (int i = 1; i <= m; i++) {
        int u = inf.readInt(), v = inf.readInt();
        ll w = inf.readInt();
        ll c = inf.readInt();
        g[u].push_back({v, w, i});
        g[v].push_back({u, w, i});
        ecost[i] = c;
    }

    // internal baseline: original shortest den->watering route (no corridors cleared)
    ll B = dijkstra(vector<char>(m + 1, 0));
    if (B == LLONG_MAX || B <= 0) quitf(_fail, "bad instance: B=%lld", B);

    // ---- read & validate participant's cleared-corridor set ----
    int r = ouf.readInt(0, m, "r");
    vector<char> removed(m + 1, 0);
    ll spent = 0;
    for (int i = 0; i < r; i++) {
        int idx = ouf.readInt(1, m, "edgeIndex");
        if (removed[idx]) quitf(_wa, "corridor %d cleared more than once", idx);
        removed[idx] = 1;
        spent += ecost[idx];
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    if (spent > C) quitf(_wa, "clearing cost %lld exceeds budget %lld", spent, C);

    ll F = dijkstra(removed);
    if (F == LLONG_MAX) quitf(_wa, "clearing these corridors cuts den %d off from watering ground %d", s, t);

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld spent=%lld Ratio: %.6f", F, B, spent, sc / 1000.0);
    return 0;
}
