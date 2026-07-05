#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt(2, 240, "n");
    int m = inf.readInt(1, 2000, "m");
    int D = inf.readInt(0, 2, "D");

    vector<int> eu(m), ev(m);
    vector<ll> ew(m);
    for (int i = 0; i < m; i++) {
        eu[i] = inf.readInt(1, n, "u");
        ev[i] = inf.readInt(1, n, "v");
        ew[i] = inf.readInt(1, 1000000, "w");
    }

    // ---- read & validate participant fleet assignment ----
    vector<int> x(n + 1, 0);
    int c0 = 0, c1 = 0;
    for (int i = 1; i <= n; i++) {
        x[i] = ouf.readInt(0, 1, "label");
        if (x[i] == 0) c0++; else c1++;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");
    if (abs(c0 - c1) > D)
        quitf(_wa, "fleet imbalance: |%d - %d| = %d > D = %d", c0, c1, abs(c0 - c1), D);

    // participant cut weight (congestion avoided)
    ll F = 0;
    for (int i = 0; i < m; i++)
        if (x[eu[i]] != x[ev[i]]) F += ew[i];

    // internal baseline: alternating assignment x_i = i mod 2
    ll B = 0;
    for (int i = 0; i < m; i++) {
        int a = eu[i] & 1, b = ev[i] & 1;
        if (a != b) B += ew[i];
    }
    if (B <= 0) quitf(_fail, "bad instance: baseline B = %lld", B);

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
