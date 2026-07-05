#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();
    int tol = inf.readInt();

    vector<int> eu(m), ev(m);
    vector<ll> ew(m);
    for (int i = 0; i < m; i++) {
        eu[i] = inf.readInt(1, n, "u");
        ev[i] = inf.readInt(1, n, "v");
        ew[i] = inf.readInt(1, 1000000000LL, "w");
    }

    // ---- internal baseline B: parity assignment b_i = i mod 2 ----
    auto cutOf = [&](const vector<int>& side) -> ll {
        ll s = 0;
        for (int i = 0; i < m; i++)
            if (side[eu[i]] != side[ev[i]]) s += ew[i];
        return s;
    };
    vector<int> par(n + 1, 0);
    for (int i = 1; i <= n; i++) par[i] = i & 1;
    ll B = cutOf(par);
    if (B <= 0) quitf(_fail, "bad instance: baseline B=%lld not positive", B);

    // ---- read & validate participant's bowl assignment ----
    vector<int> side(n + 1, 0);
    ll c0 = 0, c1 = 0;
    for (int i = 1; i <= n; i++) {
        int b = ouf.readInt(0, 1, "bowl");
        side[i] = b;
        if (b == 0) c0++; else c1++;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    if (llabs(c0 - c1) > tol)
        quitf(_wa, "bowls unbalanced: |%lld-%lld|=%lld > tol=%d", c0, c1, llabs(c0 - c1), tol);

    ll F = cutOf(side);

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
