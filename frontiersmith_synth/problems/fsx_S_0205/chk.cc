#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, s, p, k;
struct AdjE { int to; ll w; int idx; };
vector<vector<AdjE>> g;
vector<int> tanks;

// single-source Dijkstra from s; returns full dist array (LLONG_MAX = unreachable),
// skipping removed edges.
vector<ll> dijkstra(const vector<char>& removed) {
    vector<ll> dist(n + 1, LLONG_MAX);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<pair<ll,int>>> pq;
    dist[s] = 0; pq.push({0, s});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (auto& e : g[u]) {
            if (removed[e.idx]) continue;
            ll nd = d + e.w;
            if (nd < dist[e.to]) { dist[e.to] = nd; pq.push({nd, e.to}); }
        }
    }
    return dist;
}

// F = sum over tanks of dist(s, tank); returns -1 if any tank unreachable.
ll sumF(const vector<char>& removed) {
    vector<ll> dist = dijkstra(removed);
    ll tot = 0;
    for (int u : tanks) {
        if (dist[u] == LLONG_MAX) return -1;
        tot += dist[u];
    }
    return tot;
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    n = inf.readInt();
    m = inf.readInt();
    s = inf.readInt();
    p = inf.readInt();
    k = inf.readInt();

    g.assign(n + 1, {});
    for (int i = 1; i <= m; i++) {
        int u = inf.readInt(), v = inf.readInt();
        ll w = inf.readInt();
        g[u].push_back({v, w, i});
        g[v].push_back({u, w, i});
    }
    tanks.resize(p);
    for (int i = 0; i < p; i++) tanks[i] = inf.readInt();

    // internal baseline: original total pump-to-tank travel time (no valves shut)
    ll B = sumF(vector<char>(m + 1, 0));
    if (B <= 0) quitf(_fail, "bad instance: B=%lld", B);

    // ---- read & validate participant's shut-valve set ----
    int r = ouf.readInt(0, k, "r");
    vector<char> removed(m + 1, 0);
    for (int i = 0; i < r; i++) {
        int idx = ouf.readInt(1, m, "edgeIndex");
        if (removed[idx]) quitf(_wa, "valve %d shut more than once", idx);
        removed[idx] = 1;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    ll F = sumF(removed);
    if (F < 0) quitf(_wa, "shutting these valves cuts off at least one tank from the pump");

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
