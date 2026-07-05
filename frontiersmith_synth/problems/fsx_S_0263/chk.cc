#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m;
vector<ll> w;
vector<vector<int>> adj;
vector<pair<int,int>> edges;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    n = inf.readInt(1, 5000, "n");
    ll maxM = (ll)n * (n - 1) / 2;
    m = (int)inf.readInt(0, maxM, "m");
    w.assign(n + 1, 0);
    for (int i = 1; i <= n; i++) w[i] = inf.readInt(1, 1000, "w");

    adj.assign(n + 1, {});
    edges.reserve(m);
    for (int i = 0; i < m; i++) {
        int u = inf.readInt(1, n, "u");
        int v = inf.readInt(1, n, "v");
        adj[u].push_back(v);
        adj[v].push_back(u);
        edges.push_back({u, v});
    }

    // ---- internal baseline B: index-order greedy independent set ----
    // scan 1..n, light a ride iff none of its already-lit neighbours block it.
    vector<char> blocked(n + 1, 0);
    ll B = 0;
    for (int i = 1; i <= n; i++) {
        if (blocked[i]) continue;
        B += w[i];
        blocked[i] = 1;
        for (int nb : adj[i]) blocked[nb] = 1;
    }
    if (B <= 0) quitf(_fail, "bad instance: B=%lld", B);

    // ---- read & validate participant's circuit ----
    int c = ouf.readInt(0, n, "c");
    vector<char> sel(n + 1, 0);
    for (int j = 0; j < c; j++) {
        int idx = ouf.readInt(1, n, "rideIndex");
        if (sel[idx]) quitf(_wa, "ride %d listed more than once", idx);
        sel[idx] = 1;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // independence check
    for (auto& e : edges)
        if (sel[e.first] && sel[e.second])
            quitf(_wa, "conflicting rides %d and %d both lit",
                  e.first, e.second);

    ll F = 0;
    for (int i = 1; i <= n; i++) if (sel[i]) F += w[i];

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
