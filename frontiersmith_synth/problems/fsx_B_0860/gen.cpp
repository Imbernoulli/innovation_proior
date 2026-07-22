#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Analyzer Lineup: Tips and Recalibrations"   family: dual-reset-assay-lineup
//
// n samples run back to back in a chosen order. Two INDEPENDENT error processes:
//   - carryover: a concentration DROP from the previous sample costs
//     GAP_COEF * drop, unless a tip change (budget K1) is spent on this sample.
//   - drift: DRIFT_COEF * (samples since last recalibration)^2, reset to 1 right
//     after a recalibration (budget K2).
// COUPLING: an uncorrected carryover drop (no tip change, gap>0) also multiplies
// that sample's drift term by (1+BOOST) -- so the two reset currencies interact:
// a spot where carryover already broke the sequence is a spot where drift damage
// is amplified, and a spot that already got a recalibration (cnt=1) has very
// little left to gain from ALSO getting a tip change there. Some samples are
// "stat" with a hard position deadline (must run among the first `deadline_i`
// positions), forcing interruptions to an otherwise smooth ascending-concentration
// run.
//
// PLANTED TRAP (tests 5,6,7,8,9,10): concentrations are bimodal -- a low "filler"
// band and a small set of "spike" samples with MUCH higher concentration, each
// spike marked stat with a TIGHT deadline close to its own input index (which is
// scattered across the whole range, not just the front). Sorting samples purely
// by ascending concentration (the obvious carryover-killing heuristic) would push
// every spike to the very end -- violating its deadline by a wide margin. Any
// feasible order must instead interleave spikes back near their own scattered
// index, breaking the ascending run in MANY places at once. A solver that treats
// tip-budget and recal-budget allocation as two independent top-K picks (ignore
// the (1+BOOST) coupling, space recalibrations evenly) leaves real savings on the
// table versus one that places both resets jointly at the breaks.
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int n, K1, K2, GAP_COEF, DRIFT_COEF, BOOST;
    int cminF, cmaxF, cminS = 0, cmaxS = 0;
    double statFrac = 0.0, spikeFrac = 0.0;
    int slack = 10;
    bool trap = false;
    bool needle = false; (void)needle;

    switch (testId) {
        case 1: // tiny sanity (deliberately mild: real differentiation is on the trap tests)
            n = 8; K1 = 1; K2 = 1; GAP_COEF = 3; DRIFT_COEF = 1; BOOST = 2;
            statFrac = 0.35; cminF = 1; cmaxF = 100; slack = 4; break;
        case 2: // small generic (mild)
            n = 40; K1 = 2; K2 = 2; GAP_COEF = 4; DRIFT_COEF = 2; BOOST = 2;
            statFrac = 0.2; cminF = 1; cmaxF = 1000; slack = 10; break;
        case 3: // generic (mild)
            n = 150; K1 = 8; K2 = 3; GAP_COEF = 1; DRIFT_COEF = 2; BOOST = 2;
            statFrac = 0.2; cminF = 1; cmaxF = 4000; slack = 15; break;
        case 4: // generic (mild)
            n = 400; K1 = 25; K2 = 8; GAP_COEF = 3; DRIFT_COEF = 2; BOOST = 4;
            statFrac = 0.2; cminF = 1; cmaxF = 6000; slack = 20; break;
        case 5: // TRAP
            n = 600; K1 = 14; K2 = 8; GAP_COEF = 2; DRIFT_COEF = 2; BOOST = 50;
            trap = true; spikeFrac = 0.20; cminF = 1; cmaxF = 2000;
            cminS = 4000; cmaxS = 9000; slack = 3; break;
        case 6: // TRAP
            n = 900; K1 = 16; K2 = 14; GAP_COEF = 2; DRIFT_COEF = 2; BOOST = 30;
            trap = true; spikeFrac = 0.20; cminF = 1; cmaxF = 2000;
            cminS = 4500; cmaxS = 10000; slack = 3; break;
        case 7: // TRAP
            n = 1400; K1 = 14; K2 = 16; GAP_COEF = 2; DRIFT_COEF = 2; BOOST = 30;
            trap = true; spikeFrac = 0.12; cminF = 1; cmaxF = 3000;
            cminS = 5000; cmaxS = 11000; slack = 3; break;
        case 8: // TRAP / needle
            n = 2000; K1 = 25; K2 = 12; GAP_COEF = 1; DRIFT_COEF = 4; BOOST = 30;
            needle = true; trap = true; spikeFrac = 0.08; cminF = 1; cmaxF = 4000;
            cminS = 6000; cmaxS = 15000; slack = 3; break;
        case 9: // TRAP, large scale
            n = 3000; K1 = 30; K2 = 22; GAP_COEF = 2; DRIFT_COEF = 2; BOOST = 30;
            trap = true; spikeFrac = 0.08; cminF = 1; cmaxF = 3500;
            cminS = 5500; cmaxS = 13000; slack = 3; break;
        case 10: // TRAP, largest scale, fills the constraint envelope
            n = 4000; K1 = 26; K2 = 26; GAP_COEF = 1; DRIFT_COEF = 5; BOOST = 40;
            trap = true; spikeFrac = 0.08; cminF = 1; cmaxF = 4000;
            cminS = 6000; cmaxS = 16000; slack = 3; break;
        default:
            n = 20; K1 = 2; K2 = 2; GAP_COEF = 2; DRIFT_COEF = 1; BOOST = 2;
            statFrac = 0.2; cminF = 1; cmaxF = 500; slack = 8; break;
    }

    vector<ll> conc(n + 1);
    vector<int> isStat(n + 1, 0);
    vector<int> deadline(n + 1, n);

    for (int i = 1; i <= n; i++) {
        if (trap) {
            if (rnd.next(0.0, 1.0) < spikeFrac) {
                conc[i] = rnd.next(cminS, cmaxS);
                isStat[i] = 1;
                deadline[i] = min(n, i + rnd.next(0, slack));
            } else {
                conc[i] = rnd.next(cminF, cmaxF);
                isStat[i] = 0;
                deadline[i] = n;
            }
        } else {
            conc[i] = rnd.next(cminF, cmaxF);
            if (rnd.next(0.0, 1.0) < statFrac) {
                isStat[i] = 1;
                deadline[i] = min(n, i + rnd.next(0, slack));
            } else {
                isStat[i] = 0;
                deadline[i] = n;
            }
        }
    }

    printf("%d %d %d %d %d %d\n", n, K1, K2, GAP_COEF, DRIFT_COEF, BOOST);
    for (int i = 1; i <= n; i++) {
        printf("%lld %d %d\n", conc[i], isStat[i], deadline[i]);
    }
    return 0;
}
