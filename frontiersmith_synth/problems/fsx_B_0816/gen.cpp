#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Generator for "Route a Pipetting Protocol to Fill a Plate"  (serial-dilution-routing)
//
// Emits: W M Vmax Vcap stepCost stockAccessCost maxOps
//        then W lines: c_i Vreq_i
//
// Difficulty ladder (testId 1..10): W grows 6 -> 150.
// CLUSTER / TRAP test ids (>=3 of 10): target concentrations are drawn from a
//   small set of cluster centers (with jitter) instead of uniformly, so a
//   per-well direct-fill approach (the obvious "greedy" recipe) pays a full
//   stock-access touch for every well despite massive redundancy, while a
//   shared-intermediate router can serve a whole cluster from one built well.
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;

    ll W = 6 + (ll)llround(f * 144.0);      // 6 .. 150
    ll M = W + 20;                          // generous scratch-well headroom
    ll Vmax = 60 + (ll)rnd.next(0, 40);     // 60..100
    ll avgVreqLo = Vmax * 3, avgVreqHi = Vmax * 8;
    ll Vcap = Vmax * 90;                    // ~ fan-out of ~15 wells per instance
    ll stepCost = rnd.next(1, 2);
    ll stockAccessCost = rnd.next(38, 48);
    ll maxOps = 60 * (W + M) + 2000;

    // trap test ids: clustered target concentrations
    bool clustered = (testId == 3 || testId == 5 || testId == 6 || testId == 8 || testId == 9);

    vector<ll> c(W + 1), Vreq(W + 1);

    if (clustered){
        int ncl = rnd.next(2, 4);
        vector<ll> centers(ncl);
        for (int k = 0; k < ncl; k++) centers[k] = rnd.next(50, 950);
        for (ll i = 1; i <= W; i++){
            ll base = centers[rnd.next(0, ncl - 1)];
            ll jit = rnd.next(-10, 10);
            ll v = base + jit;
            if (v < 0) v = 0;
            if (v > 1000) v = 1000;
            c[i] = v;
            Vreq[i] = rnd.next(avgVreqLo, avgVreqHi);
        }
    } else {
        for (ll i = 1; i <= W; i++){
            c[i] = rnd.next(0, 1000);
            Vreq[i] = rnd.next(avgVreqLo, avgVreqHi);
        }
    }

    printf("%lld %lld %lld %lld %lld %lld %lld\n", W, M, Vmax, Vcap, stepCost, stockAccessCost, maxOps);
    for (ll i = 1; i <= W; i++) printf("%lld %lld\n", c[i], Vreq[i]);
    return 0;
}
