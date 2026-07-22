#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Robotic Telescope Night Plan"  (generator)  family: slew-drift-observation-tour
//
// A robotic telescope starts at sky coordinate 0 with pointing drift 0. It slews to
// targets and observes them. Slewing a distance d ADDS k*d to the drift
// (motion-proportional-drift) and costs d time. Observing target i at drift D yields
// v_i * acc[min(D, Dmax)] (value-weighted-accuracy), where acc[] is a steep,
// decreasing accuracy profile given in the input. A recalibration resets drift to 0
// but costs R time; at most K recalibrations are allowed and everything must fit in a
// night time budget Tb (limited-recalibration).
//
// PLANTED STRUCTURE (labels are NOT given to the solver -- coordinates are shuffled):
//   Targets come in a few dense high-value CLUSTERS (each a tight coordinate window,
//   large per-target values) scattered among cheap FILLER targets. Crucially some of
//   the most valuable clusters sit near the FAR end of the sky (coordinate ~ P), so a
//   monotone left-to-right sweep reaches them at MAXIMUM accumulated drift -> their
//   accuracy collapses to the floor -> their huge value is wasted.
//
//   Accuracy is a spendable currency the tour itself creates: the yield-optimal plan
//   BURNS drift on cheap filler and RECALIBRATES right before a high-value cluster so
//   that cluster is observed at near-zero drift -- co-designing routing and reset
//   placement. There are more clusters than recalibrations (C > K), so the resets must
//   be allocated to the clusters where a fresh calibration buys the most value.
//
//   TRAP: the "shortest tour" (coordinate sweep) spreads its resets/drift uniformly and
//   observes the far jackpot clusters at collapsed accuracy; the value-greedy "rush to
//   the biggest cluster first" arrives there with drift = k*(huge distance) and also
//   collapses. Only aligning resets to high-value clusters wins.
//
// Output format (matches statement):
//   line1:  N P k K R Tb SCALE Dmax
//   next N: p_i v_i          (distinct coordinate in [1,P], value)
//   then :  Dmax+1 integers  acc[0..Dmax]  (nonincreasing, acc[0]=SCALE, acc[Dmax]>=1)
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;

    const ll SCALE = 1000;
    ll k = 1;

    ll P = 300 + (ll)llround(f * 7000.0);          // 300 .. 7300  sky span
    ll Dmax = P;                                    // max meaningful drift
    int C = 3 + (int)llround(f * 10.0);             // 3 .. 13 clusters
    int K = 2 + (int)llround(f * 5.0);              // 2 .. 7 recalibrations (< C)
    if (K >= C) K = C - 1;
    ll R = 40;                                      // recalibration time cost
    int W = 6;                                       // cluster coordinate window width

    // ---- build clusters ----
    struct Cl { ll center; ll val; int sz; };
    vector<Cl> cls;
    // spread centers across [W+2, P-W-2]; force one jackpot near the far end.
    for (int c = 0; c < C; c++){
        ll center;
        if (c == 0) center = P - W - 2;                       // FAR jackpot cluster
        else center = (ll)(W + 2) + rnd.next(0LL, max(1LL, P - 2*(W+2)));
        int sz = 5 + rnd.next(0, 7);                           // 5 .. 12 targets
        ll val;
        if (c == 0) val = 1200 + rnd.next(0, 400);             // jackpot per-target value
        else val = 120 + rnd.next(0, 900);                     // 120 .. 1020
        cls.push_back({center, val, sz});
    }

    // ---- lay out coordinates (all distinct) ----
    set<ll> used;
    struct Tg { ll p, v; };
    vector<Tg> tg;
    for (auto &c : cls){
        int placed = 0, guard = 0;
        while (placed < c.sz && guard < 10000){
            guard++;
            ll p = c.center + rnd.next(0, W);
            if (p < 1) p = 1; if (p > P) p = P;
            if (used.count(p)) continue;
            used.insert(p);
            // small per-target jitter so within-cluster values differ a little
            ll v = c.val + rnd.next(0, 30);
            tg.push_back({p, v});
            placed++;
        }
    }
    int clusterTargets = (int)tg.size();

    int fillerCount = 5 + (int)llround(f * 1000.0);            // 5 .. 1005
    int fguard = 0;
    for (int i = 0; i < fillerCount; ){
        fguard++;
        if (fguard > 20000000) break;
        ll p = 1 + rnd.next(0LL, P - 1);
        if (used.count(p)) continue;
        used.insert(p);
        ll v = 1 + rnd.next(0, 5);                             // cheap filler 1..6
        tg.push_back({p, v});
        i++;
    }

    int N = (int)tg.size();

    // ---- night budget: enough for a near-monotone sweep + K resets + modest reorder,
    //      but NOT for large detours plus all resets (couples routing & recalibration).
    ll Tb = P + K * R + (ll)(4 * C * W + 120);

    // ---- accuracy profile: steep exponential decay, integer, nonincreasing ----
    vector<ll> acc(Dmax + 1);
    double alpha = 4.0 / (double)Dmax;                          // acc[Dmax]=1000*e^-4~=18 (floor); steep enough that resets pay, gentle enough to leave score headroom above strong
    ll prev = SCALE + 1;
    for (ll d = 0; d <= Dmax; d++){
        ll a = (ll)llround((double)SCALE * exp(-alpha * (double)d));
        if (a < 1) a = 1;
        if (a > prev) a = prev;                                 // enforce nonincreasing
        acc[d] = a;
        prev = a;
    }
    acc[0] = SCALE;

    // ---- shuffle target listing so solver cannot read structure from order ----
    for (int i = N - 1; i > 0; i--) swap(tg[i], tg[rnd.next(0, i)]);

    // ---- emit ----
    printf("%d %lld %lld %d %lld %lld %lld %lld\n", N, P, k, K, R, Tb, SCALE, Dmax);
    for (auto &t : tg) printf("%lld %lld\n", t.p, t.v);
    // accuracy table, 20 per line
    for (ll d = 0; d <= Dmax; d++){
        printf("%lld", acc[d]);
        if (d % 20 == 19 || d == Dmax) printf("\n"); else printf(" ");
    }
    return 0;
}
