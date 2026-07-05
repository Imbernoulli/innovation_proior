// testlib checker/scorer: budget-constrained max-weight independent set (vineyard irrigation)
#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    // ---- read the test input ----
    int n = inf.readInt();
    int m = inf.readInt();
    long long W = inf.readLong();

    vector<long long> w(n + 1), d(n + 1);
    for (int i = 1; i <= n; i++) w[i] = inf.readLong();
    for (int i = 1; i <= n; i++) d[i] = inf.readLong();

    vector<int> eu(m), ev(m);
    for (int e = 0; e < m; e++) {
        eu[e] = inf.readInt();
        ev[e] = inf.readInt();
    }

    // ---- internal baseline B = best single plot (always feasible, d_i <= W) ----
    long long B = 0;
    for (int i = 1; i <= n; i++)
        if (d[i] <= W) B = max(B, w[i]);
    if (B <= 0) B = 1;  // safety

    // ---- read participant output, validate feasibility strictly ----
    int c = ouf.readInt(0, n, "c");
    vector<char> chosen(n + 1, 0);
    long long F = 0, usedWater = 0;
    for (int k = 0; k < c; k++) {
        int idx = ouf.readInt(1, n, "plot");
        if (chosen[idx]) quitf(_wa, "plot %d listed more than once", idx);
        chosen[idx] = 1;
        F += w[idx];
        usedWater += d[idx];
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing data after the plot list");

    // budget constraint
    if (usedWater > W)
        quitf(_wa, "water budget exceeded: used %lld > W=%lld", usedWater, W);

    // runoff (independence) constraint
    for (int e = 0; e < m; e++) {
        if (chosen[eu[e]] && chosen[ev[e]])
            quitf(_wa, "runoff conflict: plots %d and %d both irrigated", eu[e], ev[e]);
    }

    // ---- score ----
    double sc = min(1000.0, 100.0 * (double)F / (double)max(1LL, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld water=%lld/%lld Ratio: %.6f",
          F, B, usedWater, W, sc / 1000.0);
    return 0;
}
