#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int T = inf.readInt();
    ll W = inf.readLong();
    int q = inf.readInt();
    ll K = inf.readInt();

    vector<int> avail(T + 1), g(T + 1), sc(T + 1), dc(T + 1);
    for (int t = 1; t <= T; t++) {
        avail[t] = inf.readInt();
        g[t] = inf.readInt();
        sc[t] = inf.readInt();
        dc[t] = inf.readInt();
    }

    // ---- internal baseline B: on-demand for the first ceil(W/q) slots, one run ----
    ll h = (W + (ll)q - 1) / (ll)q;
    if (h < 1) h = 1;
    if (h > T) quitf(_fail, "bad instance: h=%lld > T=%d (q*T<W)", h, T);
    ll B = K; // one startup for the contiguous on-demand block
    for (ll t = 1; t <= h; t++) B += dc[(int)t];
    if (B <= 0) quitf(_fail, "bad instance: B=%lld", B);

    // ---- read & validate participant schedule ----
    ll sacks = 0, cost = 0, segs = 0;
    bool prevActive = false;
    for (int t = 1; t <= T; t++) {
        int a = ouf.readInt(0, 2, "action");
        bool active;
        if (a == 1) {
            if (!avail[t]) quitf(_wa, "Spot used at slot %d but grid is unavailable", t);
            sacks += g[t];
            cost += sc[t];
            active = true;
        } else if (a == 2) {
            sacks += q;
            cost += dc[t];
            active = true;
        } else {
            active = false;
        }
        if (active && !prevActive) segs++;
        prevActive = active;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    if (sacks < W)
        quitf(_wa, "insufficient flour: milled %lld < required %lld", sacks, W);

    ll F = cost + K * segs;
    if (F < 1) F = 1; // feasible schedules always have F>=1 (at least one active slot)

    double sc_out = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc_out / 1000.0, "OK F=%lld B=%lld segs=%lld Ratio: %.6f", F, B, segs, sc_out / 1000.0);
    return 0;
}
