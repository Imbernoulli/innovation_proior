#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();
    vector<ll> w(n + 1);
    ll B = 0; // internal baseline: value of the single best target (always feasible)
    for (int i = 1; i <= n; i++) {
        w[i] = inf.readInt();
        B = max(B, w[i]);
    }
    vector<int> eu(m), ev(m);
    for (int e = 0; e < m; e++) {
        int u = inf.readInt(1, n);
        int v = inf.readInt(1, n);
        eu[e] = u; ev[e] = v;
    }
    if (B <= 0) quitf(_fail, "bad instance: B=%lld", B);

    // ---- read & validate participant's chosen docket ----
    int r = ouf.readInt(0, n, "r");
    vector<char> chosen(n + 1, 0);
    for (int i = 0; i < r; i++) {
        int idx = ouf.readInt(1, n, "targetIndex");
        if (chosen[idx]) quitf(_wa, "target %d scheduled more than once", idx);
        chosen[idx] = 1;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // ---- feasibility: no conflict edge inside the chosen set ----
    for (int e = 0; e < m; e++) {
        if (chosen[eu[e]] && chosen[ev[e]])
            quitf(_wa, "targets %d and %d conflict but are both scheduled", eu[e], ev[e]);
    }

    // ---- objective: total scheduled science value ----
    ll F = 0;
    for (int i = 1; i <= n; i++) if (chosen[i]) F += w[i];

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
