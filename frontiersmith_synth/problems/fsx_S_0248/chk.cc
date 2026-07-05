#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();

    vector<int> eu(m), ev(m);
    vector<ll> ew(m);
    for (int i = 0; i < m; i++) {
        eu[i] = inf.readInt();
        ev[i] = inf.readInt();
        ew[i] = inf.readInt();
    }

    // ---- internal baseline B: reference balanced split, racks 1..n/2 -> loop 0,
    // racks n/2+1..n -> loop 1. Split strength of that assignment.
    vector<int> base(n + 1);
    for (int i = 1; i <= n; i++) base[i] = (i <= n / 2) ? 0 : 1;
    ll B = 0;
    for (int i = 0; i < m; i++)
        if (base[eu[i]] != base[ev[i]]) B += ew[i];
    if (B <= 0) quitf(_fail, "bad instance: reference bisection strength B=%lld", B);

    // ---- read & validate the participant's loop assignment ----
    vector<int> a(n + 1);
    int ones = 0;
    for (int i = 1; i <= n; i++) {
        a[i] = ouf.readInt(0, 1, "loop_i");
        ones += a[i];
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");
    if (ones != n / 2)
        quitf(_wa, "unbalanced loops: %d racks on red loop, expected exactly %d", ones, n / 2);

    // ---- objective F: total strength of couplings split across the two loops ----
    ll F = 0;
    for (int i = 0; i < m; i++)
        if (a[eu[i]] != a[ev[i]]) F += ew[i];

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
