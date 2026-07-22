#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// gen.cpp -- Toroidal Sidon Beacon Placement.
// testId is a difficulty/structure ladder. Tests 3,4,6,8 are PLANTED: H*W is chosen to
// equal a perfect-difference-set modulus N = n^2+n+1 for a prime-power n, with H,W coprime
// (so Z_H x Z_W ~= Z_N via CRT and a Singer/cyclic difference set folds cleanly onto the
// torus). Those tests also ELEVATE a structured cluster of cells (a full row, or an
// arithmetic-progression subset for the degenerate 1-row torus) to a mildly higher weight
// band: a myopic single-pass "add the heaviest still-feasible cell" greedy will burn most of
// its budget on this mutually-conflicting cluster and plateau well short of the algebraic
// ceiling n+1 that an exhaustive shift/multiplier search over the cyclic difference set can
// reach. Other tests are generic random tori (no planted algebraic ceiling) so the largest
// test also fills the declared size envelope.
int main(int argc, char *argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    int idx = min(max(testId, 1), 10);

    struct Spec { int H, W; long long lnum, lden; bool trap; int baseLo, baseHi, trapLo, trapHi; };
    // lambda = lnum/lden
    Spec specs[10] = {
        /*1*/  {3, 4,   1, 2,  false,  1, 50,   0, 0},
        /*2*/  {5, 6,   2, 5,  false,  1, 300,  0, 0},
        /*3*/  {3, 7,   3, 5,  true,   1, 300,  450, 750},   // N=21=3*7, n=4,k=5 (mild trap)
        /*4*/  {1, 31,  1, 2,  true,   1, 300,  450, 750},   // N=31, n=5,k=6 (degenerate 1-row, strong trap)
        /*5*/  {6, 7,   1, 2,  false,  1, 400,  0, 0},
        /*6*/  {3, 19,  9, 20, true,   1, 400,  600, 900},   // N=57=3*19, n=7,k=8 (strong trap)
        /*7*/  {7, 8,   9, 20, false,  1, 300,  0, 0},
        /*8*/  {7, 13,  1, 2,  true,   1, 500,  700, 1000},  // N=91=7*13, n=9,k=10 (strong trap)
        /*9*/  {7, 9,   2, 5,  false,  1, 400,  0, 0},
        /*10*/ {7, 11,  2, 5,  false,  1, 400,  0, 0},
    };
    Spec sp = specs[idx - 1];
    int H = sp.H, W = sp.W;

    vector<vector<long long>> w(H, vector<long long>(W));
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
            w[i][j] = rnd.next(sp.baseLo, sp.baseHi);

    if (sp.trap) {
        if (H > 1) {
            // structured trap: elevate the ENTIRE row 0 -- a size-W cluster whose cells share
            // many pairwise differences of the form (0, d), so a naive descending-weight pass
            // rejects most of them once a couple are accepted.
            for (int j = 0; j < W; j++) w[0][j] = rnd.next(sp.trapLo, sp.trapHi);
        } else {
            // degenerate 1-row torus: elevate an arithmetic progression (step 3) within the
            // single row -- the same "shared-difference cluster" trap in 1D.
            for (int j = 0; j < W; j += 3) w[0][j] = rnd.next(sp.trapLo, sp.trapHi);
        }
    }

    printf("%d %d\n", H, W);
    printf("%lld %lld\n", sp.lnum, sp.lden);
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) printf("%lld%c", w[i][j], j + 1 == W ? '\n' : ' ');
    }
    return 0;
}
