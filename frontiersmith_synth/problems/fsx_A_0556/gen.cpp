#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Frozen Tables, Severed Cables"  (generator)  family: static-routing-under-link-death
//
// We emit an undirected connected network. The core is a set of "double-rail
// ladders" hung off a common root r (node 1). Ladder k has L layers; layer i has
// two routers a_i, b_i. Rails a_{i-1}-a_i and b_{i-1}-b_i carry the shortest paths
// toward r, so BFS distance of both a_i and b_i is exactly i.
//
// Every layer carries a RUNG a_i-b_i (same-distance link). SOME layers additionally
// carry DIAGONALS a_i-b_{i-1}, b_i-a_{i-1} (a strictly-closer second neighbor).
//
//   * A rung-only layer: a_i's ONLY strictly-closer neighbor is a_{i-1}. The obvious
//     loop-free-alternate (a strictly-closer backup) does NOT exist -> the naive
//     backup rule leaves a_i unprotected, so cutting rail edge (a_{i-1},a_i) strands
//     the whole upper rail. The INSIGHT is that the SAME-distance rung neighbor b_i
//     has a rail route to r that avoids a_i, so it is a valid (longer) escape --
//     a property of the forwarding DAG, not of path length.  <-- the trap layers.
//
//   * A diagonal layer: a_i has a strictly-closer second neighbor b_{i-1}, so even
//     the naive path-length backup protects it. These make the recipe beat do-nothing.
//
// A small pendant blob is attached to r by a single BRIDGE. Cutting that bridge
// strands the pendant from everyone -- an unavoidable floor that no table can beat,
// keeping the score ceiling open above the reference solutions.
//
// Output:  n m ; then m lines "u v".
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // (K ladders, L layers) ladder chosen to grow size but keep n<=250, m<=700.
    int Karr[11] = {0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
    int Larr[11] = {0, 3, 6, 6,10,10,14,14,18,18,22};
    int K = Karr[testId];
    int L = Larr[testId];
    int pend = (testId == 1) ? 2 : 3;   // pendant blob size (floor knob)

    int r = 1;
    int next = 2;                        // next free node id
    set<pair<int,int>> E;
    auto add = [&](int u, int v){
        if (u == v) return;
        if (u > v) swap(u, v);
        E.insert({u, v});
    };

    // Layer type governs who can protect a rail cut at that layer:
    //   diagonal layer (a strictly-closer 2nd neighbour) -> the naive LFA recipe
    //     protects it, so cutting its rail edge costs greedy almost nothing.
    //   rung-only layer (same-distance escape only) -> only the arc-disjoint
    //     insight protects it.
    // The LOW layers (largest sub-rails hanging above them = highest-damage cuts)
    // are made diagonal so the recipe fixes its own worst case; the UPPER layers
    // are rung-only so the insight is what pushes the worst case down further.
    int diagThresh = max(1, (int)llround(0.50 * L));
    for (int k = 0; k < K; k++){
        vector<int> A(L + 1), B(L + 1);
        for (int i = 1; i <= L; i++){ A[i] = next++; B[i] = next++; }
        // layer 1 hangs off the root
        add(r, A[1]); add(r, B[1]);
        add(A[1], B[1]);                 // rung at layer 1
        for (int i = 2; i <= L; i++){
            add(A[i-1], A[i]);           // rail A
            add(B[i-1], B[i]);           // rail B
            add(A[i], B[i]);             // rung (every layer -> 2-edge-connected)
            // low band diagonal, high band rung-only, with a little jitter at the
            // boundary for per-test divergence.
            int thr = diagThresh + rnd.next(-1, 1);
            if (i <= thr){               // diagonal layer: strictly-closer 2nd nbr
                add(A[i], B[i-1]);
                add(B[i], A[i-1]);
            }
        }
        // a few random same-or-forward chords for realism / divergence (never
        // create a strictly-closer shortcut that trivializes the trap layers:
        // connect within a layer band).
        int chords = rnd.next(0, L / 4);
        for (int c = 0; c < chords; c++){
            int i = rnd.next(2, L);
            int j = min(L, i + rnd.next(0, 2));
            // link a_i to a_j / b_j (i<=j) : forward or same, not strictly closer
            if (rnd.next(0, 1)) add(A[i], (rnd.next(0,1) ? A[j] : B[j]));
            else                add(B[i], (rnd.next(0,1) ? A[j] : B[j]));
        }
    }

    // ---- pendant blob attached by a single bridge to r ----
    // internal 2-edge-connected (a cycle) so within-pendant routing is fine;
    // the bridge is the sole point of failure -> unavoidable floor.
    vector<int> P;
    for (int i = 0; i < pend; i++) P.push_back(next++);
    add(r, P[0]);                        // THE bridge
    if (pend == 1){
        // nothing else
    } else if (pend == 2){
        add(P[0], P[1]);
    } else {
        for (int i = 0; i < pend; i++) add(P[i], P[(i+1) % pend]); // cycle
    }

    int n = next - 1;

    // emit
    vector<pair<int,int>> edges(E.begin(), E.end());
    printf("%d %d\n", n, (int)edges.size());
    // shuffle edge order deterministically so the input isn't structurally obvious
    shuffle(edges.begin(), edges.end());
    for (auto &e : edges) printf("%d %d\n", e.first, e.second);
    return 0;
}
