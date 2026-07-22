#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Generator -- family: taxiway-crossing-waves
//
// We emit N aircraft that must cross an active runway through periodic slots
// (open when t mod P < W, one lane => at most W per slot = a "wave").  Pushing
// back blocks the gate alley for d ticks, so alley-mates are staggered.
//
// PLANTED TRAP STRUCTURE (the obvious FCFS-by-release + earliest-fit heuristic
// lands FAR from a weight-aware wave-packing strategy):
//   * BURSTS: clusters of aircraft released at nearly the same tick contend for
//     the same scarce slots -- they must be split across several waves, and WHICH
//     aircraft ride the early waves (heavy) vs late (light) dominates weighted
//     delay.  FCFS ignores weight -> heavy aircraft spill to late waves.
//   * SHARED ALLEYS: burst aircraft share a few alleys, so pushback blocking
//     forces staggering; the *order* you push them (heavy first) is the upstream
//     lever that assembles the right wave.  FCFS pushes in release/index order.
//   * HEAVY-TAILED WEIGHTS: a few very heavy aircraft among many light ones make
//     mis-ordering expensive.
//
// Regimes by testId (1 tiny sanity ... 10 full envelope); >=3 are traps.
// Output:  "N P W d"  then N lines "rel tau w alley".
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;

    // ---- global slot / alley parameters ----
    int N, P, W, d, A;
    int regime; // 0 tiny, 1 spread-uniform, 2 burst-mixed(TRAP), 3 alley-burst(TRAP),
                // 4 oversub(TRAP), 5 skew-needle
    switch (testId) {
        case 1:  regime = 0; break;                 // tiny sanity
        case 2:  regime = 1; break;                 // spread uniform (easy baseline)
        case 3:  regime = 2; break;                 // TRAP burst-mixed
        case 4:  regime = 3; break;                 // TRAP alley-burst
        case 5:  regime = 4; break;                 // TRAP oversubscribed
        case 6:  regime = 5; break;                 // skew needle
        case 7:  regime = 1; break;                 // spread uniform larger
        case 8:  regime = 2; break;                 // TRAP burst-mixed larger
        case 9:  regime = 3; break;                 // TRAP alley-burst larger + skew
        default: regime = 4; break;                 // TRAP oversubscribed full envelope
    }

    N = 12 + (int)llround(f * 388.0);               // 12 .. 400
    P = 16 + (int)llround(f * 24.0);                // 16 .. 40
    W = 2  + (int)llround(f * 3.0);                 // 2  .. 5
    if (W >= P) W = P - 1;
    d = 3  + (int)llround(f * 9.0);                 // 3  .. 12

    if (regime == 0) { N = 12; P = 12; W = 3; d = 4; }

    // number of alleys: many for spread, few for alley-burst
    if      (regime == 1) A = max(4, N / 6);        // spread over many alleys
    else if (regime == 3) A = 2 + (int)(f * 2);     // 2..4 alleys (heavy blocking)
    else if (regime == 4) A = max(3, N / 20);       // oversubscribed, few alleys
    else                  A = max(3, N / 12);
    if (A < 1) A = 1;

    // release window R controls contention: capacity over [0,R] ~ (R/P)*W.
    // choose R so that N noticeably exceeds capacity in trap regimes.
    double frac;
    if      (regime == 1) frac = 0.75;              // roomy
    else if (regime == 4) frac = 0.32;              // heavily oversubscribed
    else                  frac = 0.50;
    ll R = (ll)llround((double)N * P / (double)W * frac);
    if (R < P) R = P;

    // number of bursts (for burst/oversub regimes)
    int nBursts;
    if      (regime == 2) nBursts = max(2, N / 25);
    else if (regime == 4) nBursts = max(2, N / 40);
    else                  nBursts = 0;

    vector<ll> burstCenter;
    for (int b = 0; b < nBursts; b++)
        burstCenter.push_back(rnd.next((ll)0, R));

    // weight sampler
    auto sampleWeight = [&](int reg)->int{
        if (reg == 1) return rnd.next(1, 50) + rnd.next(0, 30);  // near-uniform, mild
        if (reg == 5) {                                          // needle: mostly light
            if (rnd.next(0, 11) == 0) return rnd.next(700, 1000);
            return rnd.next(1, 25);
        }
        // burst / alley / oversub: heavy-tailed
        int r = rnd.next(0, 9);
        if (r < 6) return rnd.next(1, 40);
        if (r < 9) return rnd.next(100, 400);
        return rnd.next(600, 1000);
    };

    printf("%d %d %d %d\n", N, P, W, d);
    for (int i = 0; i < N; i++) {
        ll rel;
        if (nBursts > 0) {
            int b = rnd.next(0, nBursts - 1);
            ll jit = rnd.next((ll)0, (ll)max(1, (int)(P / 2)));   // tight cluster
            rel = burstCenter[b] + jit;
            if (rel < 0) rel = 0;
        } else {
            rel = rnd.next((ll)0, R);
        }
        int tau = (regime == 0) ? rnd.next(1, 4) : rnd.next(3, 80);
        int w   = sampleWeight(regime);
        int alley = rnd.next(0, A - 1);
        printf("%lld %d %d %d\n", rel, tau, w, alley);
    }
    return 0;
}
