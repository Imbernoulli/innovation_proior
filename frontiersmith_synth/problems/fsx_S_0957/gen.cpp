#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// ---- fsx_S_0957: Facade tiling that never repeats ----
// Wang-tile catalog: 16 physical tiles = every combination of
//   N,S in {10,11} (vertical boundary colors)  x  E,W in {0,1} (horizontal boundary colors).
// Because N,S range independently over {10,11} and E,W independently over {0,1},
// exactly 4 of the 16 tiles are "self-loop" tiles (N==S and E==W) that can legally
// repeat forever in every direction -- the easy periodic groove. The catalog is
// presented to the solver in a testId-seeded shuffled order, and we make sure the
// smallest-indexed tile after shuffling is NOT a self-loop tile, so a naive
// left-to-right "smallest index that satisfies the local constraint" solver has to
// hunt for a while before it (robustly, by a small-state-space pigeonhole argument)
// settles into a short horizontal cycle.

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- size ladder: tiny sanity -> wide "needle" -> largest adversarial ----
    static const int Rtab[11] = {0, 6, 8, 10, 12, 16, 18, 20, 24, 8, 32};
    static const int Wtab[11] = {0, 12, 16, 24, 40, 64, 100, 140, 180, 260, 320};
    int R = Rtab[testId];
    int W = Wtab[testId];

    // ---- build the 16-tile catalog: (N,E,S,W) ----
    struct Tile { int N, E, S, W; };
    vector<Tile> tiles;
    for (int n : {10, 11})
        for (int e : {0, 1})
            for (int s : {10, 11})
                for (int w : {0, 1})
                    tiles.push_back({n, e, s, w});
    // tiles.size() == 16

    // ---- shuffle tile presentation order (Fisher-Yates with testlib rnd) ----
    int T = (int)tiles.size();
    for (int i = T - 1; i > 0; i--) {
        int j = rnd.next(i + 1);
        swap(tiles[i], tiles[j]);
    }

    // ---- fixup: ensure tile[0] (smallest index a naive solver sees first) is
    // NOT a self-loop tile (N==S && E==W). Swap it with the first non-self-loop
    // tile found later in the list. Since exactly 12 of 16 tiles are not
    // self-loop, this always succeeds. ----
    auto isSelfLoop = [](const Tile& t) { return t.N == t.S && t.E == t.W; };
    if (isSelfLoop(tiles[0])) {
        for (int i = 1; i < T; i++) {
            if (!isSelfLoop(tiles[i])) { swap(tiles[0], tiles[i]); break; }
        }
    }

    printf("%d %d %d\n", R, W, T);
    for (auto& t : tiles)
        printf("%d %d %d %d\n", t.N, t.E, t.S, t.W);

    return 0;
}
