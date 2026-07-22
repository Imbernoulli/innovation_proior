// TIER: trivial
// The checker's own internal baseline: a plain spiral-matrix traversal from the
// entrance (0,0) into the center. It satisfies the hard unicursal/coverage
// constraint cleanly but makes no attempt at symmetry, turn-rhythm or
// depth-oscillation -- reproduces B exactly -> ratio ~= 0.1.
#include "common.h"

int main() {
    int n; long long wSym, wTurn, wOsc;
    readInput(cin, n, wSym, wTurn, wOsc);
    vector<Cell> path = spiralPath(n);
    printMaze(cout, path, n);
    return 0;
}
