#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, s, q, k;
struct AdjE { int to; ll w; int idx; };
vector<vector<AdjE>> g;

// Single-source shortest paths from s, skipping removed edges.
static vector<ll> dijkstra(const vector<char>& removed) {
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

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    n = inf.readInt();
    m = inf.readInt();
    s = inf.readInt();
    q = inf.readInt();
    k = inf.readInt();

    g.assign(n + 1, {});
    for (int i = 1; i <= m; i++) {
        int u = inf.readInt(), v = inf.readInt();
        ll w = inf.readInt();
        g[u].push_back({v, w, i});
        g[v].push_back({u, w, i});
    }
    vector<int> term(q);
    for (int i = 0; i < q; i++) term[i] = inf.readInt();

    // internal baseline: total foraging effort with no corridors destroyed
    vector<char> none(m + 1, 0);
    vector<ll> d0 = dijkstra(none);
    ll B = 0;
    for (int t : term) {
        if (d0[t] == LLONG_MAX) quitf(_fail, "bad instance: terminal %d unreachable in base graph", t);
        B += d0[t];
    }
    if (B <= 0) quitf(_fail, "bad instance: baseline B=%lld", B);

    // ---- read & validate participant's destroyed-corridor set ----
    int r = ouf.readInt(0, k, "r");
    vector<char> removed(m + 1, 0);
    for (int i = 0; i < r; i++) {
        int idx = ouf.readInt(1, m, "edgeIndex");
        if (removed[idx]) quitf(_wa, "corridor %d destroyed more than once", idx);
        removed[idx] = 1;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // ---- objective under the participant's interdiction ----
    vector<ll> d1 = dijkstra(removed);
    ll F = 0;
    for (int t : term) {
        if (d1[t] == LLONG_MAX)
            quitf(_wa, "forage patch %d is disconnected from hive %d", t, s);
        F += d1[t];
    }

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
