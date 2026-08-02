// gen.cpp -- Splinter Cairns (fsx_A_1365)
// Prints one test file: T misere duels, each a sum of M cairns (r_i, n_i).
// Every duel is filtered (via the tame-game misere theorem) to be a genuine
// first-player win, so scoring is purely about move quality, never luck.
#include "testlib.h"
#include <vector>
#include <cstdio>
using namespace std;

static int grundy(int r, int n) { return n % (r + 1); }

// Conway's tame-sum misere theorem: given the per-cairn Grundy numbers, is the
// position a P-position (the player about to move LOSES under perfect misere play)?
static bool theoremIsP(const vector<int>& gs) {
    bool anyWild = false;
    for (int g : gs) if (g >= 2) { anyWild = true; break; }
    if (anyWild) {
        int x = 0;
        for (int g : gs) x ^= g;
        return x == 0;
    }
    int c1 = 0;
    for (int g : gs) if (g == 1) c1++;
    return (c1 % 2) == 1;
}

struct Cfg { int M; int cap; double trap; int T; };

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    static const Cfg ladder[11] = {
        {0,0,0,0},                 // unused (1-indexed)
        {2,  180, 0.02,  20},
        {2,  400, 0.05,  30},
        {3,  800, 0.08,  50},
        {3, 1200, 0.35,  60},
        {4, 1800, 0.45,  80},
        {4, 2500, 0.65, 100},
        {5, 3200, 0.75, 120},
        {5, 4000, 0.80, 140},
        {6, 4600, 0.85, 160},
        {6, 5000, 0.90, 200},
    };
    Cfg cfg = ladder[testId];

    printf("%d\n", cfg.T);
    for (int t = 0; t < cfg.T; t++) {
        vector<int> rs(cfg.M), ns(cfg.M);
        // Resample until: not all-zero, and genuinely a first-player win (N-position).
        for (int attempt = 0; attempt < 500; attempt++) {
            for (int i = 0; i < cfg.M; i++) {
                int r = rnd.next(2, 5);
                int n;
                if (rnd.next(0.0, 1.0) < cfg.trap) {
                    // trap pool: force this cairn's Grundy number into {0,1}.
                    // Values with grundy(r,v) == residue are exactly
                    // {residue, residue+(r+1), residue+2(r+1), ...} -- pick
                    // the residue then a random multiple, closed form (no
                    // O(cap) scan needed).
                    int residue = rnd.next(0, 1);
                    int maxK = (cfg.cap - residue) / (r + 1);  // >= 0 since cap >= 4 > residue
                    int k = rnd.next(0, maxK);
                    n = residue + k * (r + 1);
                } else {
                    n = rnd.next(0, cfg.cap);
                }
                rs[i] = r; ns[i] = n;
            }
            bool anyNonzero = false;
            for (int v : ns) if (v > 0) anyNonzero = true;
            if (!anyNonzero) continue;
            vector<int> gs(cfg.M);
            for (int i = 0; i < cfg.M; i++) gs[i] = grundy(rs[i], ns[i]);
            if (!theoremIsP(gs)) break;  // N-position: accept
            // else resample
        }
        printf("%d\n", cfg.M);
        for (int i = 0; i < cfg.M; i++) printf("%d %d\n", rs[i], ns[i]);
    }
    return 0;
}
