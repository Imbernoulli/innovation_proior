// TIER: greedy
// The obvious first idea: the spiral is "boring" (all turns identical handedness), so
// patch it with a single-objective local search that ADDS TURN VARIETY (a small
// grid-adjacency-preserving 2-opt pass that only ever accepts a move because it raises
// turn-rhythm entropy -- with a bare "don't net-regress" safety rail, nothing more).
// It never DECIDES to help symmetry or oscillation, and has no way to tell whether a
// turn-improving splice is quietly trading those away, since on a spiral skeleton
// almost every available 2-opt splice crosses a ring boundary (a single ring is only
// 1 cell wide, so it has essentially no internal slack) -- meaning symScore/oscScore
// move as an unplanned side effect of chasing turnScore, not a deliberate result. It
// does not realize a fully symmetric wall set is provably impossible here either (the
// entrance corner forces the other 3 D4-image corners to also be degree-1 dead ends,
// contradicting "exactly two endpoints"). This is the "recipe" tier: it optimizes
// exactly one of the three composed mechanisms and is blind to the other two.
#include "common.h"

int main() {
    int n; long long wSym, wTurn, wOsc;
    readInput(cin, n, wSym, wTurn, wOsc);
    vector<Cell> path = spiralPath(n);
    auto image = buildImageTable(n);
    localSearch(path, n, image, wSym, wTurn, wOsc, /*window=*/40, /*sweeps=*/3, /*useComposite=*/false);
    printMaze(cout, path, n);
    return 0;
}
