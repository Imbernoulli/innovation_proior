#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();
    ll L = inf.readLong();
    ll U = inf.readLong();

    vector<ll> p(n + 1);
    ll T = 0;
    for (int i = 1; i <= n; i++) { p[i] = inf.readLong(); T += p[i]; }

    vector<int> eu(m), ev(m);
    vector<ll> ew(m);
    for (int i = 0; i < m; i++) {
        eu[i] = inf.readInt(1, n);
        ev[i] = inf.readInt(1, n);
        ew[i] = inf.readLong();
    }

    // ---- internal balance-only baseline construction (cut-oblivious) ----
    // scan venues in index order, assign each to the currently-lighter cohort
    // (ties -> Cohort B / label 0). Always satisfies the band by construction.
    vector<int> base(n + 1, 0);
    ll sA = 0, sB = 0;
    for (int i = 1; i <= n; i++) {
        if (sA < sB) { base[i] = 1; sA += p[i]; }   // put on lighter side A
        else         { base[i] = 0; sB += p[i]; }   // tie or B lighter -> B
    }
    if (sA < L || sA > U)
        quitf(_fail, "internal baseline out of band: sA=%lld L=%lld U=%lld", sA, L, U);

    ll B = 0;
    for (int i = 0; i < m; i++)
        if (base[eu[i]] != base[ev[i]]) B += ew[i];
    if (B <= 0) quitf(_fail, "bad instance: baseline cut B=%lld", B);

    // ---- read & validate participant's cohort labels ----
    vector<int> c(n + 1);
    ll popA = 0;
    for (int i = 1; i <= n; i++) {
        c[i] = ouf.readInt(0, 1, "label");
        if (c[i] == 1) popA += p[i];
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    if (popA < L || popA > U)
        quitf(_wa, "Cohort A population %lld outside band [%lld, %lld]", popA, L, U);

    ll F = 0;
    for (int i = 0; i < m; i++)
        if (c[eu[i]] != c[ev[i]]) F += ew[i];

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
