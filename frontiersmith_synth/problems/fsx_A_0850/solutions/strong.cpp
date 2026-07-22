// TIER: strong
// The insight: reformulate construction as a local search over the JOINT weighted
// objective, using the two structural facts a naive builder misses:
//   (1) [orbit-counting bound] the entrance (0,0) is a corner; every non-identity D4
//       symmetry maps corners to corners, so a perfectly symmetric wall set would
//       force the OTHER three corners to also be corridor-degree-1 dead ends -- but
//       "exactly two degree-1 cells" (entrance + center) is a hard requirement. Hence
//       symScore=1.0 is provably unreachable; the reachable optimum is bounded away
//       from it by a handful of necessarily-asymmetric edges near the corners/seams,
//       not by zero. A search that directly re-scores the full D4 orbit table after
//       each candidate move can chase this bound instead of chasing symmetry blindly.
//   (2) [ring-depth invariant] on the plain spiral every ring is Chebyshev-equidistant
//       from the center, so depth is *exactly* piecewise-constant -> the oscillation
//       term is identically 0 for ANY pure ring-by-ring spiral, regardless of
//       handedness. Only a 2-opt move that threads across a ring boundary (grid-
//       adjacent path positions from DIFFERENT rings) can create excess depth
//       variation, so the search must actively accept ring-crossing reversals whenever
//       they raise the composite -- exactly what optimizing the joint score (not just
//       turn-entropy) discovers on its own.
// The local search below accepts a candidate move iff it improves the FULL composite
// (sym+turn+osc jointly, per the input weights), with a wider window and more sweeps
// than the greedy tier -- balancing all three mechanisms instead of one.
#include "common.h"

int main() {
    int n; long long wSym, wTurn, wOsc;
    readInput(cin, n, wSym, wTurn, wOsc);
    vector<Cell> path = spiralPath(n);
    auto image = buildImageTable(n);
    localSearch(path, n, image, wSym, wTurn, wOsc, /*window=*/150, /*sweeps=*/20, /*useComposite=*/true);
    printMaze(cout, path, n);
    return 0;
}
