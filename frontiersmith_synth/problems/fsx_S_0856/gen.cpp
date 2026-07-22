#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Each forbidden pattern is (w,x,y,z): a 2x2 window
//   row r  , col c : w  x
//   row r+1, col c : y  z
// is illegal anywhere in the output grid.
//
// Every forbidden pattern used by this generator has z=1 (bottom-right raised).
// Consequently every pattern with z=0 -- including (0,0,0,0) and (1,0,1,0) --
// is NEVER forbidden. This guarantees two things simultaneously:
//   (a) the row pattern "1000...0" repeated on every row is always feasible,
//       giving a positive, always-feasible checker baseline;
//   (b) a column-by-column left-to-right fill can ALWAYS complete a row by
//       falling back to 0 whenever raising a cell would violate a window,
//       so the "obvious" greedy fill never gets structurally stuck.
struct Pat { int w, x, y, z; };

// Two hand-curated forbidden-window families (both include (1,1,1,1) so an
// all-ones grid is always illegal). Each was picked, and VERIFIED by
// exhaustive simulation, so that the best STATIC (single row repeated)
// pattern is weak, a period-3 CYCLE of rows realizes a substantially higher
// density -- the subshift capacity / transfer-matrix trap the problem is
// built on -- while EVERY plausible "obvious" fill strategy lands well short
// of it: (a) a left-to-right, top-to-bottom cell fill (raise unless it would
// complete a forbidden window), and (b) ANY one-step-lookahead row-to-row
// fill (from an unconstrained first row, at each step continue with a row
// achieving the maximum raised-thread count among the immediately-compatible
// rows -- for ANY choice of tie-break among equally-good next rows). Earlier
// forbidden-window choices were found (by simulation, and independently by
// review) to let some particular tie-break of strategy (b) accidentally
// rediscover a near-optimal cycle on some instances; these two families were
// re-selected by explicitly simulating the BEST-CASE outcome over every
// possible tie-break of strategy (b) (not just one arbitrary tie-break) and
// keeping only families where that best case still falls well short of the
// true transfer-matrix optimum, at every grid width 4 <= C <= 9 used below.
static const vector<Pat> FAM_A = { {0,0,0,1},{0,0,1,1},{0,1,0,1},{1,1,0,1},{1,1,1,1} };
static const vector<Pat> FAM_D = { {0,1,0,1},{1,1,0,1},{1,1,1,1} };

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    struct Case { int C; int R; const vector<Pat>* fam; };
    // ladder: tiny example -> small -> trap cases -> large trap cases filling the envelope.
    static const Case table[11] = {
        {},                       // unused index 0
        {4,    6,   &FAM_A},      // 1: tiny worked example
        {5,   40,   &FAM_A},      // 2: small
        {6,   80,   &FAM_A},      // 3: small
        {5,  150,   &FAM_A},      // 4: TRAP
        {6,  300,   &FAM_A},      // 5: TRAP
        {7,  600,   &FAM_A},      // 6: TRAP
        {8, 1200,   &FAM_A},      // 7: TRAP - widening gap at larger C
        {8, 2500,   &FAM_D},      // 8: TRAP - large, diversity family
        {9, 6000,   &FAM_A},      // 9: TRAP - large scale
        {9,15000,   &FAM_D},      // 10: TRAP - largest, diversity family, fills constraint envelope
    };

    int C = table[testId].C;
    int R = table[testId].R + rnd.next(0, 4); // deterministic small jitter given testId
    const vector<Pat>& fam = *table[testId].fam;
    int K = (int)fam.size();

    printf("%d %d %d\n", R, C, K);
    for (auto& p : fam)
        printf("%d %d %d %d\n", p.w, p.x, p.y, p.z);

    return 0;
}
