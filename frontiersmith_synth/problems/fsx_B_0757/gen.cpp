#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Generator for "Coral Skin Against the Current".
// Emits one line: W H S M Fx Fy
// testId is a difficulty/structure ladder: 1 tiny (example scale) up to 10 large,
// with several TRAP cases (tight overhang S=1 + a strongly one-sided current +
// wide grid, so anchoring the trunk near the boundary opposite the current matters
// a lot and an un-anchored / symmetric grower loses heavily) and a NEEDLE case
// (only one direction carries real bonus).

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    long long W, H, S, M, Fx, Fy;

    auto emit = [&]() {
        printf("%lld %lld %lld %lld %lld %lld\n", W, H, S, M, Fx, Fy);
    };

    if (testId == 1) {
        // tiny, example scale
        W = 8; H = 6; S = 1; M = 10; Fx = 1; Fy = 0;
    } else if (testId == 2) {
        // small, diagonal current
        W = 10; H = 8; S = 2; M = 16; Fx = -1; Fy = 1;
    } else if (testId == 3) {
        // TRAP: tight overhang, strong one-sided current, moderate width
        W = 20; H = 14; S = 1; M = 60; Fx = 5; Fy = 0;
    } else if (testId == 4) {
        // TRAP: current points left, wide grid -- anchoring near the right edge
        // matters; a scan-order/symmetric grower wastes width on the wrong side
        W = 30; H = 16; S = 1; M = 90; Fx = -6; Fy = 1;
    } else if (testId == 5) {
        // NEEDLE: only "up-and-right" carries real bonus (Fy small but Fx big),
        // everything else near-flat -- must specifically bias branches that way
        W = 24; H = 20; S = 2; M = 110; Fx = 6; Fy = 1;
    } else if (testId == 6) {
        // TRAP: dense budget forces a genuine pack-vs-branch trade-off
        W = 18; H = 18; S = 2; M = 260; Fx = 3; Fy = -2;
    } else if (testId == 7) {
        // medium-large, diagonal current, moderate overhang
        W = 28; H = 24; S = 3; M = 300; Fx = -3; Fy = 3;
    } else if (testId == 8) {
        // TRAP: tightest overhang (S=1) + widest grid + strongest pure horizontal
        // current -- reaching the flow-favorable extreme needs a long, correctly
        // anchored chain; naive local placement can't recover the wasted width
        W = 40; H = 30; S = 1; M = 400; Fx = 6; Fy = 0;
    } else if (testId == 9) {
        // large scale, randomized (via testlib's testId-seeded rnd), moderate S
        W = 34; H = 34; S = 3;
        M = 800;
        do { Fx = rnd.next(-6, 6); Fy = rnd.next(-6, 6); } while (Fx == 0 && Fy == 0);
    } else {
        // testId == 10: max scale, fills the constraint envelope
        W = 40; H = 40; S = 4;
        M = W * H; // dense: use the full envelope
        do { Fx = rnd.next(-6, 6); Fy = rnd.next(-6, 6); } while (Fx == 0 && Fy == 0);
    }

    emit();
    return 0;
}
