#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// gen.cpp -- Codebook Shuffle Against Residue Spies.
// testId is a difficulty ladder: p grows from 101 (tiny, testId=1) to 4999 (testId=10).
// Weight (spy-strength) profiles vary from near-uniform (idx 1-2, everything cheap to
// decorrelate) through mild/moderate skew (idx 3-7) to DENSE heavy weighting on idx 8-10
// (every channel 1..p-1 carries a large weight, no "free" low-weight channels to ignore) --
// this is the trap regime: with p large AND every one of the ~p channels binding, a
// swap-based local search (which must re-balance ~p simultaneous max-constraints) cannot
// converge within a realistic compute budget, while a closed-form monomial-map search
// (x -> x^k mod p) decorrelates every channel simultaneously by construction.
int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    int idx = min(max(testId, 1), 10);

    int ps[10] = {101, 251, 503, 997, 1499, 1999, 2503, 3001, 3989, 4999};
    int p = ps[idx - 1];

    printf("%d\n", p);
    for (int d = 1; d <= p - 1; d++) {
        int w;
        if (idx <= 2) {
            // near-uniform, all cheap
            w = rnd.next(1, 5);
        } else if (idx <= 4) {
            // mild skew: mostly cheap, ~15% expensive spikes
            w = (rnd.next(0, 99) < 15) ? rnd.next(30, 80) : rnd.next(1, 5);
        } else if (idx <= 7) {
            // moderate skew: mostly cheap, ~25-30% expensive spikes
            int thresh = (idx == 7) ? 30 : 25;
            w = (rnd.next(0, 99) < thresh) ? rnd.next(50, 150) : rnd.next(1, 10);
        } else {
            // DENSE / heavy trap: every channel is significant, no free lunch
            int lo = (idx == 10) ? 30 : 20;
            int hi = (idx == 10) ? 300 : 200;
            w = rnd.next(lo, hi);
        }
        printf("%d%c", w, d == p - 1 ? '\n' : ' ');
    }
    return 0;
}
