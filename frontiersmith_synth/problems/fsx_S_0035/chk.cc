#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m;
vector<ll> w;
vector<vector<int>> adj;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    n = inf.readInt();
    m = inf.readInt();
    w.assign(n + 1, 0);
    for (int i = 1; i <= n; i++) w[i] = inf.readInt();
    adj.assign(n + 1, {});
    for (int i = 0; i < m; i++) {
        int u = inf.readInt(), v = inf.readInt();
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // ---- internal baseline B: index-order greedy independent set ----
    // scan 1..n; add vertex iff none of its neighbours already added.
    vector<char> inB(n + 1, 0);
    ll B = 0;
    for (int u = 1; u <= n; u++) {
        bool ok = true;
        for (int v : adj[u]) if (inB[v]) { ok = false; break; }
        if (ok) { inB[u] = 1; B += w[u]; }
    }
    if (B <= 0) quitf(_fail, "bad instance: baseline B=%lld", B);

    // ---- read & validate participant's chosen set ----
    int r = ouf.readInt(0, n, "r");
    vector<char> chosen(n + 1, 0);
    vector<int> pick;
    pick.reserve(r);
    for (int i = 0; i < r; i++) {
        int idx = ouf.readInt(1, n, "installation");
        if (chosen[idx]) quitf(_wa, "installation %d chosen more than once", idx);
        chosen[idx] = 1;
        pick.push_back(idx);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // ---- feasibility: chosen set must be independent (no conflict edge inside) ----
    for (int u : pick)
        for (int v : adj[u])
            if (chosen[v])
                quitf(_wa, "installations %d and %d conflict but are both on the tour", u, v);

    // ---- objective F = total value of chosen set ----
    ll F = 0;
    for (int u : pick) F += w[u];

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
