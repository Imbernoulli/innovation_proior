#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// -----------------------------------------------------------------------------
// "One Broken Mat: Auspicious Tatami Court Layout"  (generator)
// family: tatami-court-layout
//
// A court is an H x W grid (H, W both odd, so it has a unique centre cell).
// Some cells may be permanently blocked pillars ('#'); pillars are always placed
// as POINT-SYMMETRIC pairs (if (r,c) is a pillar, so is (H-1-r,W-1-c)), and the
// true centre cell is NEVER a pillar. Everything else is free ('.').
//
// The generator does not compute a reference tiling itself (the checker scores
// ANY partial packing of dominoes ('H'/'V' mats) plus exactly one 1x1 mat ('S'),
// not a forced exact cover), so producing a feasible instance is trivial: the
// room shape alone (free/blocked layout) is the whole test case.
//
// Ladder across testId 1..10: growing size (small sanity case -> a large case
// that fills the stated envelope), increasing pillar density (0 -> sparse ->
// moderate), and both square and elongated rectangles for shape variety. All
// pillar layouts are point-symmetric pairs so a solver that recognises the
// centre-defect / mirror-symmetry mechanic always has a clean, exploitable
// global symmetry to aim for; a solver that just runs a naive dense pass never
// notices it (this is the trap: running-bond / row-scan tilings satisfy the
// no-four-corner rule trivially with near-zero orientation entropy and near-
// chance mirror-symmetry).
// -----------------------------------------------------------------------------

static int H, W;
static vector<string> grid;

static bool inb(int r, int c) { return r >= 0 && r < H && c >= 0 && c < W; }

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // size ladder (both dims always odd)
    static const int Hs[11] = {0, 15, 19, 25, 27, 35, 41, 61, 81, 101, 151};
    static const int Ws[11] = {0, 15, 27, 25, 41, 35, 91, 61, 81, 131, 151};
    // pillar density (approx fraction of cells attempted as symmetric pillar pairs)
    static const double dens[11] = {0, 0.000, 0.000, 0.010, 0.015, 0.000,
                                     0.020, 0.000, 0.025, 0.030, 0.015};

    H = Hs[testId];
    W = Ws[testId];
    grid.assign(H, string(W, '.'));

    int cr = (H - 1) / 2, cc = (W - 1) / 2;

    int targetPairs = (int)llround(dens[testId] * (double)H * (double)W / 2.0);
    int placed = 0, attempts = 0;
    while (placed < targetPairs && attempts < targetPairs * 40 + 200) {
        attempts++;
        int r = rnd.next(0, H - 1);
        int c = rnd.next(0, W - 1);
        int mr = H - 1 - r, mc = W - 1 - c;
        if (r == cr && c == cc) continue;              // never block the centre
        if (grid[r][c] != '.' || grid[mr][mc] != '.') continue;
        // keep pillars from clustering into a 2x2 solid block (would just shrink
        // the room locally without adding interesting structure)
        int neighbourBlocked = 0;
        for (int dr = -1; dr <= 1; dr++)
            for (int dc = -1; dc <= 1; dc++) {
                int nr = r + dr, nc = c + dc;
                if (inb(nr, nc) && grid[nr][nc] == '#') neighbourBlocked++;
            }
        if (neighbourBlocked > 0) continue;
        grid[r][c] = '#';
        grid[mr][mc] = '#';
        placed++;
    }
    grid[cr][cc] = '.';  // guarantee (defensively; we never touched it above)

    printf("%d %d\n", H, W);
    for (int r = 0; r < H; r++) printf("%s\n", grid[r].c_str());
    return 0;
}
