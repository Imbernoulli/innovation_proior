#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// -----------------------------------------------------------------------------
// Generator for "Dragon-Forge Radiator Sculpting".  family: fin-taper-perimeter-tradeoff
//
// Emits a hot wall segment (row 0, columns [BC,BC+BW-1] fixed at T_HOT), an
// ambient temperature T_AMB, a material budget M, and the pair of physical
// constants (K_COND, K_LOSS) that set the conduction-to-loss ratio governing
// how far heat can travel through a given cross-section before it has all
// leaked out through exposed faces. The solver must sculpt a connected
// polyomino of at most M blocks attached to the wall.
//
// Category mix across the 10 tests:
//   1    : tiny sanity (matches statement scale).
//   2-3  : general small/medium random regimes.
//   4-5  : PLANTED generous-conduction regimes (K_COND >> K_LOSS) -- a wide,
//          tapered trunk clearly captures far more heat than either a thin
//          dendrite or an oversized block; plenty of room ("scale":"small"
//          headline case, mid-size grid).
//   6-8  : TRAP harsh-bottleneck regimes (K_LOSS >> K_COND, wall width BW>=2 so
//          widening past one column is always physically capable of helping)
//          -- conduction chokes off very fast, so any one-cell-wide structure
//          (the pure "maximize exposed perimeter" instinct) leaves most of its
//          material sitting near T_AMB and shedding almost nothing, while a
//          modestly wider/tapered trunk still captures a healthy fraction of
//          the flux. (BW=1 is deliberately avoided here: with only one root
//          contact edge, ALL flux must cross that single edge regardless of
//          downstream width, so widening past the wall can only ever help via
//          a secondary effect, never via extra root throughput -- a genuinely
//          different, less interesting regime than the intended taper trade-off.)
//   9    : mixed regime, larger scale.
//   10   : largest test, fills the declared constraint envelope (H,W,M all at
//          their ceiling).
// -----------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int H, W, M, Thot, Tamb, Kcond, Kloss, BW, BC;

    auto pickBaseRange = [&](int minBW, int maxBW) {
        int lo = max(1, minBW), hi = max(lo, min(maxBW, W));
        BW = rnd.next(lo, hi);
        BC = rnd.next(0, W - BW);
    };
    auto pickBase = [&](int maxBW) { pickBaseRange(1, maxBW); };

    switch (testId) {
        case 1: // tiny sanity
            H = 5; W = 5; M = 5;
            pickBase(1);
            Thot = 300; Tamb = 0;
            Kcond = 10; Kloss = 4;
            break;
        case 2: // general small
            H = rnd.next(9, 11); W = rnd.next(9, 11); M = rnd.next(12, 16);
            pickBase(3);
            Thot = rnd.next(300, 700); Tamb = rnd.next(0, 30);
            Kcond = rnd.next(18, 32); Kloss = rnd.next(2, 4);
            break;
        case 3: // general medium
            H = rnd.next(11, 13); W = rnd.next(11, 13); M = rnd.next(18, 24);
            pickBase(4);
            Thot = rnd.next(300, 800); Tamb = rnd.next(0, 40);
            Kcond = rnd.next(16, 30); Kloss = rnd.next(2, 5);
            break;
        case 4: // PLANTED generous conduction #1
            H = rnd.next(16, 19); W = rnd.next(14, 17); M = rnd.next(36, 44);
            pickBase(6);
            Thot = rnd.next(400, 900); Tamb = rnd.next(0, 40);
            Kcond = rnd.next(35, 55); Kloss = rnd.next(1, 2);
            break;
        case 5: // PLANTED generous conduction #2
            H = rnd.next(19, 22); W = rnd.next(17, 19); M = rnd.next(50, 60);
            pickBase(5);
            Thot = rnd.next(400, 1000); Tamb = rnd.next(0, 50);
            Kcond = rnd.next(30, 50); Kloss = rnd.next(1, 2);
            break;
        case 6: // TRAP harsh bottleneck #1 -- BW>=2 so a modest widened trunk is
                // always physically able to beat a single 1-wide contact.
            H = rnd.next(21, 24); W = rnd.next(15, 17); M = rnd.next(55, 65);
            pickBaseRange(2, 4);
            Thot = rnd.next(500, 1200); Tamb = rnd.next(0, 50);
            Kcond = rnd.next(2, 4); Kloss = rnd.next(9, 16);
            break;
        case 7: // TRAP harsh bottleneck #2 -- distinct scale/coefficients, still
                // BW>=2 (a forced single-column wall makes ANY widening strictly
                // pointless regardless of K_COND/K_LOSS, since all flux must cross
                // the one root edge -- that degenerate regime is not a taper trap,
                // it is a different, uninteresting problem, so we avoid BW=1 here).
            H = rnd.next(23, 26); W = rnd.next(16, 19); M = rnd.next(65, 75);
            pickBaseRange(2, 4);
            Thot = rnd.next(500, 1300); Tamb = rnd.next(0, 50);
            Kcond = rnd.next(2, 3); Kloss = rnd.next(10, 18);
            break;
        case 8: // TRAP harsh bottleneck #3, larger budget wasted by dendrites
            H = rnd.next(25, 28); W = rnd.next(19, 22); M = rnd.next(85, 95);
            pickBaseRange(3, 4);
            Thot = rnd.next(600, 1400); Tamb = rnd.next(0, 60);
            Kcond = rnd.next(2, 4); Kloss = rnd.next(9, 20);
            break;
        case 9: // mixed, larger
            H = rnd.next(27, 30); W = rnd.next(23, 26); M = rnd.next(120, 140);
            pickBase(8);
            Thot = rnd.next(600, 1500); Tamb = rnd.next(0, 60);
            Kcond = rnd.next(20, 34); Kloss = rnd.next(2, 5);
            break;
        case 10: // largest, fills the constraint envelope
            H = 34; W = 34; M = 260;
            pickBase(10);
            Thot = rnd.next(900, 1800); Tamb = rnd.next(0, 80);
            Kcond = rnd.next(25, 45); Kloss = rnd.next(1, 3);
            break;
        default:
            H = 10; W = 10; M = 12;
            pickBase(3);
            Thot = 400; Tamb = 10;
            Kcond = 8; Kloss = 5;
            break;
    }

    if (M > H * W) M = H * W;
    if (M < 1) M = 1; // never emit a degenerate zero-budget test
    // Guarantee (by construction, not just by the range choices above) that the
    // wall is always meaningfully hotter than the ambient air, so the checker's
    // internal baseline B is always strictly positive and every ratio in [0,1]
    // is well defined -- this is a stronger, explicit invariant than just "the
    // sampled ranges happen not to overlap".
    if (Thot < Tamb + 50) Thot = Tamb + 50;

    printf("%d %d %d %d %d %d %d\n", H, W, M, Thot, Tamb, Kcond, Kloss);
    printf("%d %d\n", BW, BC);
    return 0;
}
