#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// gen.cpp -- Single-Crystal Blade Casting: seed placement for a chain of wide "bulges"
// (root / platform / airfoil bands / tip) joined by narrow geometric "waists".  testId is
// a difficulty/structure ladder: idx 1 is a small hand-scale instance, idx 10 the largest.
// Each bulge gets its own zone resistance class (0..4, cheapest to most sluggish
// solidification); each waist is deliberately 1-3 cells wide.  The seed budget K is
// sometimes LESS than the number of bulges (forces choosing which waists matter most) and
// sometimes MORE (extra seeds dumped in an open bulge are pure liability for greedy
// farthest-point placement).  LAMBDA (fill-time weight) varies per test so neither term of
// the objective F = G + LAMBDA*M dominates on every case.

struct Row { int lo, hi, zone; };

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    int idx = min(max(testId, 1), 10);

    int SEGs[10]     = {2, 3, 3, 4, 4, 5, 5, 6, 6, 7};
    int Ks[10]        = {2, 2, 3, 2, 6, 4, 3, 7, 5, 6};
    int LAMBDAs[10]   = {4, 5, 3, 8, 2, 5, 10, 2, 6, 5};
    bool noisy[10]    = {false, false, false, false, false, false, false, true, true, true};

    int SEG = SEGs[idx - 1];
    int K = Ks[idx - 1];
    int LAMBDA = LAMBDAs[idx - 1];
    bool noise = noisy[idx - 1];

    int zoneCycle[5] = {0, 2, 4, 1, 3};

    // growth factor for row length ranges by test index -> larger instances at high idx
    int lenLo = 6 + idx / 2, lenHi = 10 + idx;         // bulge row-count range
    int hwLo = 4 + (idx == 1 ? 0 : 2), hwHi = 7 + idx / 2; // bulge half-width range
    if (idx == 1) { lenLo = 4; lenHi = 6; hwLo = 3; hwHi = 4; }

    vector<Row> rows;
    int centerCol = 0;

    auto emitSegment = [&](int length, int hw, int zoneDigit) {
        for (int t = 0; t < length; t++) {
            if (rnd.next(0, 99) < 45) centerCol += (rnd.next(0, 1) ? 1 : -1);
            int lo = centerCol - hw, hi = centerCol + hw;
            rows.push_back({lo, hi, zoneDigit});
        }
    };

    for (int b = 0; b < SEG; b++) {
        int hw = rnd.next(hwLo, hwHi);
        int length = rnd.next(lenLo, lenHi);
        int zoneDigit = zoneCycle[b % 5];
        emitSegment(length, hw, zoneDigit);
        if (b + 1 < SEG) {
            int whw = rnd.next(1, (idx == 1 ? 1 : 3));
            int wlen = rnd.next(2, (idx == 1 ? 2 : 4));
            int wzone = zoneCycle[(b + 1) % 5];
            emitSegment(wlen, whw, wzone);
        }
    }

    int gmin = INT_MAX, gmax = INT_MIN;
    for (auto& r : rows) { gmin = min(gmin, r.lo); gmax = max(gmax, r.hi); }
    int W = (gmax - gmin + 1) + 2;
    int H = (int)rows.size() + 2;

    vector<string> grid(H, string(W, '#'));
    for (int i = 0; i < (int)rows.size(); i++) {
        int printedLo = rows[i].lo - gmin + 1;
        int printedHi = rows[i].hi - gmin + 1;
        for (int c = printedLo; c <= printedHi; c++) {
            int zd = rows[i].zone;
            if (noise && rnd.next(0, 99) < 10) {
                zd += (rnd.next(0, 1) ? 1 : -1);
                zd = max(0, min(4, zd));
            }
            grid[i + 1][c] = char('0' + zd);
        }
    }

    printf("%d %d %d\n", H, W, K);
    printf("%d\n", LAMBDA);
    for (int r = 0; r < H; r++) printf("%s\n", grid[r].c_str());
    return 0;
}
