#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();
    int C = inf.readInt();

    vector<int> eu(m), ev(m);
    vector<ll> ew(m);
    ll B = 0; // do-nothing (all niche 1): every channel monochromatic -> sum of weights
    for (int i = 0; i < m; i++) {
        eu[i] = inf.readInt(1, n);
        ev[i] = inf.readInt(1, n);
        ew[i] = inf.readInt(1, (ll)1e9);
        B += ew[i];
    }
    if (B <= 0) quitf(_fail, "bad instance: B=%lld (no positive-weight channels)", B);

    // ---- read & validate participant's niche assignment ----
    vector<int> lab(n + 1);
    for (int i = 1; i <= n; i++)
        lab[i] = ouf.readInt(1, C, "niche");
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // ---- objective: total monochromatic conflict weight ----
    ll F = 0;
    for (int i = 0; i < m; i++)
        if (lab[eu[i]] == lab[ev[i]]) F += ew[i];

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
