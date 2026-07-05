#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt(1, 2500, "n");
    int m = inf.readInt(1, 20000, "m");

    vector<ll> w(m);
    vector<vector<int>> lits(m);
    for (int c = 0; c < m; c++) {
        w[c] = inf.readInt(1, 1000000000, "w");
        int L = inf.readInt(1, n, "L");
        lits[c].resize(L);
        for (int j = 0; j < L; j++) {
            int t = inf.readInt(-n, n, "lit");
            if (t == 0) quitf(_fail, "clause %d has a zero literal", c + 1);
            lits[c][j] = t;
        }
    }

    // ---- internal baseline B: weight satisfied by the all-low-power plan (x_i = 0) ----
    // a positive literal +i is false; a negative literal -i is true.
    // so a clause is baseline-satisfied iff it contains at least one negative literal.
    ll B = 0;
    for (int c = 0; c < m; c++) {
        bool sat = false;
        for (int l : lits[c]) if (l < 0) { sat = true; break; }
        if (sat) B += w[c];
    }
    if (B <= 0) quitf(_fail, "bad instance: baseline B=%lld", B);

    // ---- read & validate the participant assignment ----
    vector<int> x(n + 1, 0);
    for (int i = 1; i <= n; i++)
        x[i] = ouf.readInt(0, 1, "x_i");
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens (expected exactly %d values)", n);

    // ---- objective F: weight satisfied by the participant assignment ----
    ll F = 0;
    for (int c = 0; c < m; c++) {
        bool sat = false;
        for (int l : lits[c]) {
            int v = abs(l);
            if (l > 0) { if (x[v] == 1) { sat = true; break; } }
            else       { if (x[v] == 0) { sat = true; break; } }
        }
        if (sat) F += w[c];
    }

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
