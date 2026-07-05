#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int k = inf.readInt();
    int m = inf.readInt();

    vector<int> eu(m), ev(m);
    vector<ll> ew(m);
    for (int i = 0; i < m; i++) {
        eu[i] = inf.readInt();
        ev[i] = inf.readInt();
        ew[i] = inf.readInt();
    }

    // ---- internal baseline B: channel-cycling plan c_g = ((g-1) mod k) + 1 ----
    // (matches the statement + trivial solution exactly).
    auto baselineChannel = [&](int g) -> int { return ((g - 1) % k) + 1; };
    ll B = 0;
    for (int i = 0; i < m; i++)
        if (baselineChannel(eu[i]) == baselineChannel(ev[i])) B += ew[i];
    if (B <= 0) quitf(_fail, "bad instance: baseline annoyance B=%lld is non-positive", B);

    // ---- read & validate participant's channel assignment ----
    vector<int> c(n + 1);
    for (int g = 1; g <= n; g++)
        c[g] = ouf.readInt(1, k, "channel");
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after %d channels", n);

    // ---- objective F: total annoyance of monochromatic overlaps ----
    ll F = 0;
    for (int i = 0; i < m; i++)
        if (c[eu[i]] == c[ev[i]]) F += ew[i];

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
