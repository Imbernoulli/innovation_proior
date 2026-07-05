// testlib checker/scorer for "Draining the Deluge: Reservoir Junction Interdiction".
// Node-deletion shortest-path interdiction, maximization.
//   F = shortest s-t path after deleting the chosen junctions (must stay connected).
//   B = shortest s-t path in the original network (do-nothing objective, positive).
//   score = min(1.0, 0.1 * F / B).
#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, s, t;
ll K;
vector<int> cst;
vector<vector<pair<int,int>>> adj;  // adj[u] = {(v,w)}

// Dijkstra on the subgraph excluding nodes flagged in `dead`. Returns shortest
// distance s->t, or -1 if t is unreachable.
ll dijkstra(const vector<char>& dead) {
    const ll INF = (ll)4e18;
    vector<ll> dist(n + 1, INF);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<>> pq;
    dist[s] = 0;
    pq.push({0, s});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d != dist[u]) continue;
        if (u == t) return d;
        for (auto& [v, w] : adj[u]) {
            if (dead[v]) continue;
            if (d + w < dist[v]) { dist[v] = d + w; pq.push({dist[v], v}); }
        }
    }
    return dist[t] == INF ? -1 : dist[t];
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    n = inf.readInt();
    m = inf.readInt();
    s = inf.readInt();
    t = inf.readInt();
    K = inf.readLong();
    cst.assign(n + 1, 0);
    for (int i = 1; i <= n; i++) cst[i] = inf.readInt();
    adj.assign(n + 1, {});
    for (int i = 0; i < m; i++) {
        int u = inf.readInt(), v = inf.readInt(), w = inf.readInt();
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }

    // baseline B = original shortest s-t path
    vector<char> none(n + 1, 0);
    ll B = dijkstra(none);
    if (B <= 0) quitf(_fail, "internal: baseline B=%lld is non-positive", B);

    // read participant's chosen junctions
    int r = ouf.readInt(0, n, "r");
    vector<char> dead(n + 1, 0);
    ll totalCost = 0;
    for (int i = 0; i < r; i++) {
        int x = ouf.readInt(1, n, "junction");
        if (x == s) quitf(_wa, "cannot shut the reservoir s=%d", s);
        if (x == t) quitf(_wa, "cannot shut the city intake t=%d", t);
        if (dead[x]) quitf(_wa, "junction %d listed more than once", x);
        dead[x] = 1;
        totalCost += cst[x];
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing data after the junction list");

    if (totalCost > K)
        quitf(_wa, "budget exceeded: total cost %lld > K=%lld", totalCost, K);

    ll F = dijkstra(dead);
    if (F < 0) quitf(_wa, "reservoir and city intake disconnected after shutdown");

    // F >= B always (deleting nodes only lengthens the shortest path).
    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld r=%d cost=%lld/%lld Ratio: %.6f",
          F, B, r, totalCost, K, sc / 1000.0);
    return 0;
}
