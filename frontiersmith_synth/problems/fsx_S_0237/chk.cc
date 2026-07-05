#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();
    int K = inf.readInt();

    vector<int> eu(m), ev(m), ew(m), ed(m);
    ll B = 0;                        // do-nothing baseline: everyone in cohort 1
    for (int i = 0; i < m; i++) {
        eu[i] = inf.readInt();
        ev[i] = inf.readInt();
        ew[i] = inf.readInt();
        ed[i] = inf.readInt();
        B += (ll)ew[i] * ed[i];      // diff = 0 on every pairing
    }
    if (B <= 0) quitf(_fail, "bad instance: B=%lld", B);

    // ---- read & validate participant's cohort assignment ----
    vector<int> c(n + 1);
    for (int i = 1; i <= n; i++)
        c[i] = ouf.readInt(1, K, "cohort");
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // ---- objective ----
    ll F = 0;
    for (int i = 0; i < m; i++) {
        int diff = abs(c[eu[i]] - c[ev[i]]);
        int pen = ed[i] - diff;
        if (pen < 0) pen = 0;
        F += (ll)ew[i] * pen;
    }

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
