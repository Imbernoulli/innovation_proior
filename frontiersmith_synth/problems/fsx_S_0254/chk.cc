#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N, M;
ll R;
vector<ll> cost;
struct E { int to; ll w; };
vector<vector<E>> g;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    N = inf.readInt();
    M = inf.readInt();
    R = inf.readLong();

    cost.assign(N + 1, 0);
    ll B = 0; // internal baseline: cost of placing a sensor at EVERY pool (trivially feasible)
    for (int v = 1; v <= N; v++) { cost[v] = inf.readLong(); B += cost[v]; }
    if (B <= 0) quitf(_fail, "bad instance: nonpositive total cost B=%lld", B);

    g.assign(N + 1, {});
    for (int i = 0; i < M; i++) {
        int u = inf.readInt(), v = inf.readInt();
        ll w = inf.readLong();
        g[u].push_back({v, w});
        g[v].push_back({u, w});
    }

    // ---- read & validate participant's sensor set ----
    int K = ouf.readInt(0, N, "K");
    vector<char> chosen(N + 1, 0);
    vector<int> sensors;
    sensors.reserve(K);
    ll F = 0;
    for (int i = 0; i < K; i++) {
        int p = ouf.readInt(1, N, "poolIndex");
        if (chosen[p]) quitf(_wa, "pool %d chosen more than once", p);
        chosen[p] = 1;
        sensors.push_back(p);
        F += cost[p];
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // ---- feasibility: every pool must lie within channel-distance R of a sensor ----
    // multi-source bounded Dijkstra from all chosen sensors.
    const ll INF = (ll)4e18;
    vector<ll> dist(N + 1, INF);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<pair<ll,int>>> pq;
    for (int p : sensors) { dist[p] = 0; pq.push({0, p}); }
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (auto& e : g[u]) {
            ll nd = d + e.w;
            if (nd <= R && nd < dist[e.to]) { dist[e.to] = nd; pq.push({nd, e.to}); }
        }
    }
    for (int v = 1; v <= N; v++)
        if (dist[v] > R)
            quitf(_wa, "pool %d is not within radius %lld of any sensor", v, R);

    if (F <= 0) quitf(_wa, "no sensors placed but pools remain uncovered");

    // minimization score: baseline / achieved, calibrated so trivial (place everywhere) -> 0.1.
    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
