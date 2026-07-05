#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

/*
 * Checker/scorer for Apiary Yard Split (balanced minimum k-cut).
 *
 * Feasibility: n labels in [0,k-1]; each yard holds exactly n/k hives.
 * Objective  : F = total weight of records whose endpoints are in different yards (minimize).
 * Baseline B : cross-yard drift of the index-block reference split
 *              (hive i -> yard (i-1)/(n/k)); always >= 1 via max(1, .).
 * Score      : sc = min(1000, 100 * B / max(1, F)); ratio = sc/1000.
 */

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    long long m = inf.readLong();
    int k = inf.readInt();

    vector<int> eu(m), ev(m);
    vector<long long> ew(m);
    for (long long e = 0; e < m; e++) {
        eu[e] = inf.readInt();
        ev[e] = inf.readInt();
        ew[e] = inf.readInt();
    }

    int s = n / k; // hives per yard

    // read participant output: n yard labels in [0,k-1]
    vector<int> yard(n + 1);
    vector<long long> cnt(k, 0);
    for (int i = 1; i <= n; i++) {
        int y = ouf.readInt(0, k - 1, format("y[%d]", i).c_str());
        yard[i] = y;
        cnt[y]++;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after %d labels", n);

    for (int g = 0; g < k; g++) {
        if (cnt[g] != s)
            quitf(_wa, "unbalanced: yard %d has %lld hives, expected %d", g, cnt[g], s);
    }

    // participant objective F (cross-yard drift)
    long long F = 0;
    for (long long e = 0; e < m; e++)
        if (yard[eu[e]] != yard[ev[e]]) F += ew[e];

    // internal baseline B: index-block reference split, hive i -> (i-1)/s
    long long B = 0;
    for (long long e = 0; e < m; e++) {
        int a = (eu[e] - 1) / s;
        int b = (ev[e] - 1) / s;
        if (a != b) B += ew[e];
    }
    if (B <= 0) B = 1; // keep baseline positive on degenerate tiny instances

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
