#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

struct AdjE { int to; ll w; int idx; };

int n, m, P, k;
vector<vector<AdjE>> g;      // node -> outgoing edges (with edge index)
vector<int> src, dst, pr;    // demand endpoints and priorities

// Dijkstra from a single source, skipping removed edges. Returns dist vector.
static void dijkstra(int s, const vector<char>& removed, vector<ll>& dist) {
    dist.assign(n + 1, LLONG_MAX);
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
}

// Priority-weighted total lag over all demands, given a removed-edge mask.
// Returns LLONG_MAX (sentinel) if ANY demand pair is disconnected.
static ll total_lag(const vector<char>& removed) {
    // group demands by source to run one Dijkstra per distinct source
    map<int, vector<int>> bySrc;
    for (int i = 0; i < P; i++) bySrc[src[i]].push_back(i);
    ll F = 0;
    vector<ll> dist;
    for (auto& kv : bySrc) {
        dijkstra(kv.first, removed, dist);
        for (int i : kv.second) {
            if (dist[dst[i]] == LLONG_MAX) return LLONG_MAX;
            F += (ll)pr[i] * dist[dst[i]];
        }
    }
    return F;
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    n = inf.readInt();
    m = inf.readInt();
    P = inf.readInt();
    k = inf.readInt();

    g.assign(n + 1, {});
    for (int i = 1; i <= m; i++) {
        int u = inf.readInt(), v = inf.readInt();
        ll w = inf.readInt();
        g[u].push_back({v, w, i});
        g[v].push_back({u, w, i});
    }
    src.resize(P); dst.resize(P); pr.resize(P);
    for (int i = 0; i < P; i++) {
        src[i] = inf.readInt();
        dst[i] = inf.readInt();
        pr[i]  = inf.readInt();
    }

    // ---- internal baseline B: no links unplugged ----
    ll B = total_lag(vector<char>(m + 1, 0));
    if (B == LLONG_MAX || B <= 0) quitf(_fail, "bad instance: B=%lld", B);

    // ---- read & validate participant's unplugged-link set ----
    int r = ouf.readInt(0, k, "r");
    vector<char> removed(m + 1, 0);
    for (int i = 0; i < r; i++) {
        int idx = ouf.readInt(1, m, "edgeIndex");
        if (removed[idx]) quitf(_wa, "link %d unplugged more than once", idx);
        removed[idx] = 1;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // ---- objective F ----
    ll F = total_lag(removed);
    if (F == LLONG_MAX)
        quitf(_wa, "unplugging these links disconnects a monitored connection");

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
