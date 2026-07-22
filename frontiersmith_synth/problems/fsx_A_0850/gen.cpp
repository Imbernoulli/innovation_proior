#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Generator for "Veiled Fourfold Hedge" (family: hedge-maze-veiled-symmetry).
// Prints: "n wSym wTurn wOsc" on one line. n is odd (entrance=(0,0),
// center=((n-1)/2,(n-1)/2) are implicit). testId is a difficulty/weight-skew ladder,
// growing n toward the largest adversarial case. The reference greedy tier only ever
// chases turn-rhythm entropy (see solutions/greedy.cpp); oscillation-heavy weights
// (cases 5, 8, 10) are the sharpest TRAP -- greedy has no way to seek oscillation
// credit at all, while the joint (strong) search finds it, so the gap is largest
// there. Symmetry-heavy weights (cases 3, 7) are a milder trap: symScore starts high
// on a plain spiral already (most of each ring self-matches under D4) and is hard for
// ANY tier to move much further, so absolute gaps stay small there by construction --
// included for coverage of the mechanism, not as the primary trap.

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // n ladder (all odd, grows to the size cap used by the reference local search).
    static const int nList[11] = {0, 5, 7, 9, 11, 13, 15, 17, 19, 23, 27};
    int n = nList[testId];

    long long wSym, wTurn, wOsc;
    switch (testId) {
        case 1: wSym = 3; wTurn = 3; wOsc = 3; break;                 // tiny sanity, balanced
        case 2: wSym = 3; wTurn = 3; wOsc = 3; break;                 // balanced
        case 3: wSym = 7; wTurn = 1; wOsc = 1; break;                 // symmetry dominates (mild trap, see above)
        case 4: wSym = 1; wTurn = 7; wOsc = 1; break;                 // turn-entropy dominates (greedy's own axis)
        case 5: wSym = 1; wTurn = 1; wOsc = 7; break;                 // TRAP: oscillation dominates
        case 6: wSym = 2; wTurn = 2; wOsc = 2; break;                 // balanced, mid n
        case 7: wSym = 8; wTurn = 1; wOsc = 1; break;                 // symmetry dominates harder (mild trap)
        case 8: wSym = 1; wTurn = 1; wOsc = 8; break;                 // TRAP: oscillation dominates harder
        case 9: wSym = 3; wTurn = 3; wOsc = 4; break;                 // balanced, larger n
        default: wSym = 4; wTurn = 1; wOsc = 4; break;                // TRAP: sym+osc vs turn, largest n
    }
    // Small deterministic jitter (seeded by testId via testlib's rnd) so weights are
    // not exact round numbers, while keeping the intended skew pattern per case.
    wSym += rnd.next(0, 1);
    wTurn += rnd.next(0, 1);
    wOsc += rnd.next(0, 1);

    printf("%d %lld %lld %lld\n", n, wSym, wTurn, wOsc);
    return 0;
}
