#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();

    vector<ll> w(n + 1);
    ll B = 0;                         // baseline: best single site (always feasible)
    for (int i = 1; i <= n; i++) {
        w[i] = inf.readInt();
        if (w[i] > B) B = w[i];
    }
    if (B <= 0) quitf(_fail, "bad instance: nonpositive max weight");

    vector<pair<int,int>> edges(m);
    for (int i = 0; i < m; i++) {
        int u = inf.readInt(1, n, "u");
        int v = inf.readInt(1, n, "v");
        edges[i] = {u, v};
    }

    // ---- read & validate participant's independent set ----
    int c = ouf.readInt(0, n, "c");
    vector<char> chosen(n + 1, 0);
    for (int i = 0; i < c; i++) {
        int idx = ouf.readInt(1, n, "siteIndex");
        if (chosen[idx]) quitf(_wa, "site %d activated more than once", idx);
        chosen[idx] = 1;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // feasibility: no conflict edge fully inside the chosen set
    for (auto& e : edges) {
        if (chosen[e.first] && chosen[e.second])
            quitf(_wa, "activated sites %d and %d conflict", e.first, e.second);
    }

    ll F = 0;
    for (int i = 1; i <= n; i++) if (chosen[i]) F += w[i];

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
