#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// -----------------------------------------------------------------------------
// "Frustrated Wang Tiling: route the unavoidable defects"
// family: wang-frustration-tiling
//
// A Wang tile has four colored edges (W,E,N,S). We place one tile (types reusable)
// into every cell of an R x C grid. A horizontal border matches iff left.E == right.W;
// a vertical border matches iff top.S == bottom.N. Maximize matched borders.
//
// PLANTED STRUCTURE (the checker never sees these labels -- it only counts matches):
//   The "ideal" palette is a PHASE FIELD:  tile(x,y) has
//        W = x,  E = (x+1) mod P,  N = P+y,  S = P + ((y+1) mod Q).
//   With the full ideal palette the periodic assignment tile(c%P, r%Q) matches EVERY
//   border. Frustration is planted by REMOVING a small set of ideal tiles: because a
//   grid with R>=Q, C>=P visits every residue (x,y), *every* global phase offset needs
//   every ideal tile, so removing any tile makes a perfectly matched tiling IMPOSSIBLE
//   -- a fixed number of defects is forced no matter what. The task is to ROUTE those
//   forced defects (place a substitute that breaks the fewest borders) while keeping the
//   rest of the grid on the periodic field.
//
// THE TRAP (why the obvious greedy craters):  for every removed ideal tile(x,y) we add
//   a DECOY tile:  W = x,  E = POISON_H,  N = P+y,  S = POISON_V.
//   At a cell whose ideal tile was removed, the decoy is the UNIQUE tile that matches
//   BOTH the (ideal) left neighbor and the (ideal) top neighbor -- so a scanline greedy
//   that places the locally best-matching tile grabs the decoy. But the decoy's E and S
//   are poison colors no tile can match, so the rest of that row and column desynchronize
//   from the phase field: one local win, a cascade of downstream losses. The insight is
//   to treat the grid as a frustrated constraint graph, keep the global periodic field,
//   and spend the forced defects on the cheapest borders -- defect routing, not greedy
//   local matching.
//
// Output:  R C T ; then T lines  W E N S  (tile types, 0-indexed).
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int id = atoi(argv[1]);
    double f = (id - 1) / 9.0;

    int P = 3 + (int)lround(f * 13.0);          // 3..16 horizontal period
    int Q = 3 + (int)lround(f * 12.0);          // 3..15 vertical period
    if (id % 2 == 0) Q += 1;                     // incommensurate flavor
    if (Q < 3) Q = 3;

    int R = max(Q, 4 + (int)lround(f * 180.0));  // rows,  >= Q
    int C = max(P, 4 + (int)lround(f * 180.0));  // cols,  >= P

    const int POISON_H = 1000000;
    const int POISON_V = 1000001;

    // ---- forced-defect set: remove k ideal tiles (never (0,0) or (1,0)) ----
    // These make a perfectly matched tiling IMPOSSIBLE (strong stays below the cap).
    int k = 1 + (int)lround(f * 0.05 * (double)(P * Q));
    set<pair<int,int>> removed;
    int guard = 0;
    while ((int)removed.size() < k && guard < 100000){
        guard++;
        int x = rnd.next(0, P - 1), y = rnd.next(0, Q - 1);
        if ((x == 0 && y == 0) || (x == 1 && y == 0)) continue;  // reserved for baseline
        removed.insert({x, y});
    }

    // ---- trap set: PRESENT ideal tiles that ALSO get a poison decoy placed at a low
    // index. At such a cell the ideal and the decoy both match left+top (score 2), so a
    // lowest-index scanline greedy grabs the DECOY and desynchronizes the row/column;
    // the phase-field solver ignores the decoy and stays perfect there. ----
    int tt = 2 + (int)lround(f * 3.0);            // 2..5 trap tiles
    set<pair<int,int>> trap;
    guard = 0;
    while ((int)trap.size() < tt && guard < 100000){
        guard++;
        int x = rnd.next(0, P - 1), y = rnd.next(0, Q - 1);
        if ((x == 0 && y == 0) || (x == 1 && y == 0)) continue;  // keep greedy's start clean
        if (removed.count({x, y})) continue;                      // present tiles only
        trap.insert({x, y});
    }

    // ---- build the tile list. Index 0 is the CLEAN ideal tile(0,0): it is greedy's
    // lowest-index fallback when nothing matches, so greedy can re-synchronize between
    // traps (a partial recovery -- greedy is not degenerate, just far from strong).
    // Next come the DECOYS (low index, so at a trap/defect cell greedy prefers a poison
    // decoy over the equally-scoring ideal). Then the remaining ideal tiles. ----
    vector<array<int,4>> tiles;   // W,E,N,S
    tiles.push_back({0, 1 % P, P + 0, P + (1 % Q)});           // index 0: ideal tile(0,0)
    for (auto& pr : removed)
        tiles.push_back({pr.first, POISON_H, P + pr.second, POISON_V});
    for (auto& pr : trap)
        tiles.push_back({pr.first, POISON_H, P + pr.second, POISON_V});
    for (int y = 0; y < Q; y++)
        for (int x = 0; x < P; x++){
            if (x == 0 && y == 0) continue;                    // already emitted at index 0
            if (removed.count({x, y})) continue;
            tiles.push_back({x, (x + 1) % P, P + y, P + ((y + 1) % Q)});
        }

    int T = (int)tiles.size();
    printf("%d %d %d\n", R, C, T);
    for (auto& t : tiles) printf("%d %d %d %d\n", t[0], t[1], t[2], t[3]);
    return 0;
}
