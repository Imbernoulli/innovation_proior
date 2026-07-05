#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();
    vector<int> eu(m), ev(m);
    vector<long long> ew(m);
    for (int e = 0; e < m; e++) {
        eu[e] = inf.readInt();
        ev[e] = inf.readInt();
        ew[e] = inf.readInt();
    }

    // read participant output: n labels in {0,1}
    vector<int> side(n + 1);
    long long ones = 0;
    for (int i = 1; i <= n; i++) {
        int s = ouf.readInt(0, 1, format("s[%d]", i).c_str());
        side[i] = s;
        ones += s;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after %d labels", n);

    if (ones != n / 2)
        quitf(_wa, "unbalanced split: %lld sensors on station 1, expected %d", ones, n / 2);

    // participant objective
    long long F = 0;
    for (int e = 0; e < m; e++)
        if (side[eu[e]] != side[ev[e]]) F += ew[e];

    // internal baseline B: reference (id) split 1..n/2 -> 0, n/2+1..n -> 1
    long long B = 0;
    for (int e = 0; e < m; e++) {
        int a = (eu[e] <= n / 2) ? 0 : 1;
        int b = (ev[e] <= n / 2) ? 0 : 1;
        if (a != b) B += ew[e];
    }
    if (B <= 0) B = 1; // safety; generator guarantees a crossing edge

    double sc = min(1000.0, 100.0 * (double)F / (double)max(1LL, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
