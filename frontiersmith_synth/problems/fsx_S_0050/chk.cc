#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, D, P;
struct AdjE { int to; ll w; };
vector<vector<AdjE>> g;
vector<int> demandList;
vector<int> demIdOf;            // node -> demand index (1..D) or 0
vector<int> stNode, stCost, stRad;

// Dijkstra from station's node, mark every demand within its radius as covered.
// Cutoff: stop expanding nodes whose distance exceeds the radius.
void markCoverage(int stIdx, vector<char>& covered) {
    int src = stNode[stIdx];
    ll rad = stRad[stIdx];
    vector<ll> dist(n + 1, LLONG_MAX);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<pair<ll,int>>> pq;
    dist[src] = 0; pq.push({0, src});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        if (d > rad) continue;                 // beyond coverage; do not expand
        if (demIdOf[u]) covered[demIdOf[u]] = 1;
        for (auto& e : g[u]) {
            ll nd = d + e.w;
            if (nd <= rad && nd < dist[e.to]) { dist[e.to] = nd; pq.push({nd, e.to}); }
        }
    }
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    n = inf.readInt();
    m = inf.readInt();
    g.assign(n + 1, {});
    for (int i = 0; i < m; i++) {
        int u = inf.readInt(), v = inf.readInt();
        ll w = inf.readInt();
        g[u].push_back({v, w});
        g[v].push_back({u, w});
    }
    D = inf.readInt();
    demandList.resize(D);
    demIdOf.assign(n + 1, 0);
    for (int i = 0; i < D; i++) {
        demandList[i] = inf.readInt();
        demIdOf[demandList[i]] = i + 1;
    }
    P = inf.readInt();
    stNode.assign(P + 1, 0); stCost.assign(P + 1, 0); stRad.assign(P + 1, 0);
    ll B = 0;
    for (int i = 1; i <= P; i++) {
        stNode[i] = inf.readInt();
        stCost[i] = inf.readInt();
        stRad[i]  = inf.readInt();
        B += stCost[i];
    }
    if (B <= 0) quitf(_fail, "bad instance: total station cost B=%lld", B);

    // ---- read & validate participant's chosen station set ----
    int q = ouf.readInt(0, P, "q");
    vector<char> chosen(P + 1, 0);
    ll F = 0;
    for (int i = 0; i < q; i++) {
        int idx = ouf.readInt(1, P, "stationIndex");
        if (chosen[idx]) quitf(_wa, "station %d built more than once", idx);
        chosen[idx] = 1;
        F += stCost[idx];
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // ---- verify every demand block is monitored ----
    vector<char> covered(D + 1, 0);
    for (int i = 1; i <= P; i++) if (chosen[i]) markCoverage(i, covered);
    for (int i = 1; i <= D; i++)
        if (!covered[i])
            quitf(_wa, "demand block %d (node %d) is not monitored by any built station",
                  i, demandList[i - 1]);

    if (F <= 0) quitf(_wa, "no stations built but demand exists");

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
