#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m;
ll tau;
vector<int> a;
vector<int> eu, ev, ew;

ll cutOf(const vector<int>& stage) {
    ll F = 0;
    for (int e = 0; e < m; e++)
        if (stage[eu[e]] != stage[ev[e]]) F += ew[e];
    return F;
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    n = inf.readInt();
    m = inf.readInt();
    tau = inf.readLong();
    a.assign(n + 1, 0);
    for (int i = 1; i <= n; i++) a[i] = inf.readInt();
    eu.assign(m, 0); ev.assign(m, 0); ew.assign(m, 0);
    for (int e = 0; e < m; e++) {
        eu[e] = inf.readInt();
        ev[e] = inf.readInt();
        ew[e] = inf.readInt();
    }

    // ---- internal balance baseline B: input-order, always-to-lighter-stage ----
    vector<int> base(n + 1, 0);
    ll T0 = 0, T1 = 0;
    for (int i = 1; i <= n; i++) {
        if (T0 <= T1) { base[i] = 0; T0 += a[i]; }
        else          { base[i] = 1; T1 += a[i]; }
    }
    ll B = cutOf(base);
    if (B <= 0) quitf(_fail, "bad instance: baseline flow B=%lld (need > 0)", B);

    // ---- read & validate participant's stage assignment ----
    vector<int> stage(n + 1, 0);
    ll P0 = 0, P1 = 0;
    for (int i = 1; i <= n; i++) {
        int s = ouf.readInt(0, 1, "stage");
        stage[i] = s;
        if (s == 0) P0 += a[i]; else P1 += a[i];
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    ll diff = llabs(P0 - P1);
    if (diff > tau)
        quitf(_wa, "stages unbalanced: |T0-T1|=%lld > tau=%lld", diff, tau);

    ll F = cutOf(stage);

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
