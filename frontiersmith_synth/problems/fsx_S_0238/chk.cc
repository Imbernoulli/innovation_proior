#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();

    // clauses stored as list of literals
    vector<int> cw(m);
    vector<vector<int>> lit(m);
    for (int j = 0; j < m; j++) {
        cw[j] = inf.readInt();
        int k = inf.readInt();
        lit[j].resize(k);
        for (int i = 0; i < k; i++) lit[j][i] = inf.readInt();
    }

    // read participant assignment: exactly n values, each 0/1
    vector<int> x(n + 1, 0);
    for (int v = 1; v <= n; v++) {
        int b = ouf.readInt(0, 1, "x_v");
        x[v] = b;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens after the %d switch states", n);

    // internal baseline B = weight satisfied by all-OFF assignment
    long long B = 0;
    for (int j = 0; j < m; j++) {
        bool sat = false;
        for (int L : lit[j]) {
            int v = abs(L);
            int val = 0; // all-OFF
            bool litTrue = (L > 0) ? (val == 1) : (val == 0);
            if (litTrue) { sat = true; break; }
        }
        if (sat) B += cw[j];
    }
    if (B <= 0) B = 1; // safety; generator guarantees B > 0

    // participant objective F
    long long F = 0;
    for (int j = 0; j < m; j++) {
        bool sat = false;
        for (int L : lit[j]) {
            int v = abs(L);
            int val = x[v];
            bool litTrue = (L > 0) ? (val == 1) : (val == 0);
            if (litTrue) { sat = true; break; }
        }
        if (sat) F += cw[j];
    }

    double sc = min(1000.0, 100.0 * (double)F / (double)max(1LL, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
