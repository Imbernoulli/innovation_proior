#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- scale ladder ----
    static const int Ts[10] = {4, 20, 40, 60, 100, 150, 200, 250, 320, 400};
    static const int Ns[10] = {5, 200, 800, 2000, 6000, 12000, 20000, 32000, 45000, 60000};
    static const ll  Rs[10] = {20, 120, 300, 600, 1200, 2500, 5000, 12000, 30000, 100000};
    int idx = testId - 1;
    if (idx < 0) idx = 0;
    if (idx > 9) idx = 9;
    int T = Ts[idx];
    int N = Ns[idx];
    ll  R = Rs[idx];

    // Large mains price so that avoiding mains (via hoarded cheap rain) is decisive.
    ll G = rnd.next(400, 1000);

    // Two structural regimes (alternate by parity) so per-test behavior diverges, but BOTH
    // make hoarding cheap rain in the cistern valuable:
    //   SPARSE (odd testId): ~35% cheap high-catchment steps, the rest zero-catchment (so
    //     trivial/greedy pay full mains there while the storage plan serves them for free).
    //   GRADIENT (even testId): every step has catchment but the rain price rises over time,
    //     so hoarding early-cheap rain beats sourcing pricey rain late.
    bool sparse = (testId % 2 == 1);

    // Ample per-cheap-step catchment: total cheap catchment comfortably exceeds total demand.
    double avgD = 4.5;
    ll totalDemandApprox = (ll)llround(avgD * (double)N);
    ll Cmax = max<ll>(6, totalDemandApprox / max(1, T) * 5);
    if (Cmax > 100000) Cmax = 100000;

    vector<ll> f(T + 1), s(T + 1), cap(T + 1);
    for (int t = 1; t <= T; t++) {
        if (sparse) {
            if (rnd.next(0, 99) < 35) {              // cheap catchment step
                f[t]   = rnd.next(0, 40);
                s[t]   = rnd.next((ll)1, max<ll>(1, G / 40));
                cap[t] = rnd.next((ll)(Cmax * 3 / 4), (ll)Cmax);
            } else {                                 // expensive / dry step
                f[t]   = rnd.next(0, 2000);
                s[t]   = rnd.next((ll)(G / 2), (ll)G);
                cap[t] = 0;
            }
        } else {
            // rain price climbs from ~cheap early to ~mains-expensive late
            double frac = (T > 1) ? (double)(t - 1) / (double)(T - 1) : 0.0;
            ll lo = (ll)llround(1 + frac * (double)(G - 1) * 0.9);
            ll hi = (ll)llround(1 + frac * (double)(G - 1));
            if (hi < lo) hi = lo;
            f[t]   = rnd.next(0, 400);
            s[t]   = rnd.next(lo, hi);
            cap[t] = rnd.next((ll)(Cmax / 2), (ll)Cmax);
        }
        if (s[t] < 1) s[t] = 1;
        if (s[t] > G) s[t] = G;
        if (cap[t] < 0) cap[t] = 0;
        if (cap[t] > 100000) cap[t] = 100000;
    }

    // ---- beds: valid windows + demands ----
    vector<int> r(N), dl(N);
    vector<ll> d(N);
    for (int i = 0; i < N; i++) {
        int home = rnd.next(1, T);
        double roll = rnd.next(0.0, 1.0);
        int lo, hi;
        if (roll < 0.5) {                 // roomy window (can roam to cheap steps)
            lo = rnd.next(1, home);
            hi = rnd.next(home, T);
        } else {                          // tight window (nearly pinned)
            int back = rnd.next(0, 2);
            int fwd  = rnd.next(0, 2);
            lo = max(1, home - back);
            hi = min(T, home + fwd);
        }
        r[i]  = lo;
        dl[i] = hi;
        d[i]  = rnd.next(1, 8);
    }

    // shuffle bed order so input order != any natural schedule order
    vector<int> perm(N);
    iota(perm.begin(), perm.end(), 0);
    shuffle(perm.begin(), perm.end());

    printf("%d %d %lld\n", T, N, R);
    printf("%lld\n", G);
    for (int t = 1; t <= T; t++)
        printf("%lld %lld %lld\n", f[t], s[t], cap[t]);
    for (int i = 0; i < N; i++)
        printf("%d %d %lld\n", r[perm[i]], dl[perm[i]], d[perm[i]]);
    return 0;
}
