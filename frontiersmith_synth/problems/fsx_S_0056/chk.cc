#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();
    int slack = inf.readInt();

    vector<int> eu(m), ev(m);
    vector<ll> ew(m);
    for (int i = 0; i < m; i++) {
        int u = inf.readInt(1, n, "u");
        int v = inf.readInt(1, n, "v");
        ll w = inf.readInt((ll)1, (ll)1000000, "w");
        eu[i] = u; ev[i] = v; ew[i] = w;
    }

    // ---- internal baseline B: the index split (zones 1..n/2 -> 0, rest -> 1) ----
    // Always feasible (exactly balanced). Guaranteed positive by the generator.
    vector<int> base(n + 1);
    for (int i = 1; i <= n; i++) base[i] = (i <= n / 2) ? 0 : 1;
    ll B = 0;
    for (int i = 0; i < m; i++) if (base[eu[i]] != base[ev[i]]) B += ew[i];
    if (B <= 0) quitf(_fail, "bad instance: baseline B=%lld not positive", B);

    // ---- read & validate participant's labeling ----
    vector<int> lab(n + 1);
    ll s0 = 0;
    for (int i = 1; i <= n; i++) {
        int d = ouf.readInt(0, 1, "label");
        lab[i] = d;
        if (d == 0) s0++;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens (expected exactly %d labels)", n);

    ll lo = (ll)n / 2 - slack, hi = (ll)n / 2 + slack;
    if (s0 < lo || s0 > hi)
        quitf(_wa, "unbalanced: depot 0 has %lld zones, allowed window [%lld,%lld]", s0, lo, hi);

    // ---- objective F ----
    ll F = 0;
    for (int i = 0; i < m; i++) if (lab[eu[i]] != lab[ev[i]]) F += ew[i];

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
