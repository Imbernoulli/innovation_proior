#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Maximum-Weight Independent Set scorer for "Recycling Depot Routes".
//   F = total profit of the participant's selected (pairwise non-conflicting) routes.
//   B = profit of the single most-profitable route (best-single-unit baseline).
//   ratio = min(1.0, 0.1 * F / B)   (max-objective convention).

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();

    vector<ll> w(n + 1);
    ll B = 0;                              // best single route profit (baseline)
    for (int i = 1; i <= n; i++) {
        w[i] = inf.readInt();
        B = max(B, w[i]);
    }
    if (B <= 0) quitf(_fail, "bad instance: B=%lld", B);

    vector<vector<int>> g(n + 1);
    for (int i = 0; i < m; i++) {
        int u = inf.readInt(1, n, "u");
        int v = inf.readInt(1, n, "v");
        g[u].push_back(v);
        g[v].push_back(u);
    }

    // ---- read & validate participant's selected routes ----
    int c = ouf.readInt(0, n, "c");
    vector<char> chosen(n + 1, 0);
    vector<int> sel;
    sel.reserve(c);
    ll F = 0;
    for (int i = 0; i < c; i++) {
        int v = ouf.readInt(1, n, "routeIndex");
        if (chosen[v]) quitf(_wa, "route %d selected more than once", v);
        chosen[v] = 1;
        sel.push_back(v);
        F += w[v];
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // independence check: no two selected routes may conflict.
    // O(sum of degrees of selected) <= O(m).
    for (int u : sel)
        for (int v : g[u])
            if (chosen[v])
                quitf(_wa, "selected routes %d and %d conflict (not independent)", u, v);

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
