#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// -----------------------------------------------------------------------------
// "Couple oscillators just enough to synchronize"  (generator)
// family: chaotic-map-sync-coupling
//
// Instance shape (a "dumbbell"): two disjoint groups of s chaotic-logistic-map
// oscillators. WITHIN each group every pair is a candidate coupling edge
// (capacity INTRA_CAP). BETWEEN the two groups there is exactly a PERFECT
// MATCHING of s low-visibility candidate edges (capacity BRIDGE_CAP), pairing
// group-0 node i with group-1 node perm(i) for a random permutation perm
// (so a solver cannot hardcode "node i links to node s+i" -- it must find the
// cut edges from the data). The matching edges are the only path between the
// two groups, so they alone control the coupling Laplacian's spectral gap
// between the groups; the O(s^2) within-group edges cannot substitute no
// matter how many of them are funded, and a per-node coupling-degree cap of
// 1.0 means a node already busy inside its own group cannot also carry much
// of the matching load "for free".
//
// All oscillators share the SAME logistic parameter R (only initial states
// differ), so the two groups are, in isolation, generic distinct realizations
// of one chaotic map -- exactly synchronizable in principle, but only if
// enough coupling crosses the cut before the trailing-window score is read.
//
// Edges are emitted ALL group-0-internal, THEN all group-1-internal, THEN the
// matching (bridge) edges LAST -- an obvious single-pass "fill in the order
// you see them" allocator exhausts its coupling-degree budget on the very
// numerous internal edges long before it ever reaches the few (but decisive)
// matching edges.
//
// Input format:
//   N M R C T W
//   x_1 ... x_N               (initial states, 1-indexed)
//   M lines: u v cap          (candidate coupling edge, 1-indexed nodes)
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    static const int sTable[11] = {0, 4, 6, 8, 10, 14, 18, 24, 32, 45, 60};
    int s = sTable[testId];
    int N = 2 * s;

    const double R = 3.90;
    const double INTRA_CAP = 0.35;
    const double BRIDGE_CAP = 0.40;
    const double C = (double)s;
    const int T = 60;
    const int W = 20;

    vector<double> x0(N + 1);
    for (int i = 1; i <= N; i++) x0[i] = 0.02 + 0.96 * rnd.next(0.0, 1.0);

    vector<array<int,3>> edges; // u, v, capX1e6 (cap stored *1e6 as int for exact print later via /1e6)
    // group 0 internal (local 1..s)
    vector<pair<int,int>> intra0, intra1, bridge;
    for (int i = 1; i <= s - 1; i++)
        for (int j = i + 1; j <= s; j++)
            intra0.push_back({i, j});
    for (int i = 1; i <= s - 1; i++)
        for (int j = i + 1; j <= s; j++)
            intra1.push_back({s + i, s + j});

    // random permutation for the matching (anti-memorization: not identity i<->s+i)
    vector<int> perm(s);
    for (int i = 0; i < s; i++) perm[i] = i;
    for (int i = s - 1; i > 0; i--) swap(perm[i], perm[rnd.next(0, i)]);
    for (int i = 0; i < s; i++) bridge.push_back({i + 1, s + perm[i] + 1});

    // shuffle within each group for texture (group order intra0,intra1,bridge stays fixed)
    for (int i = (int)intra0.size() - 1; i > 0; i--) swap(intra0[i], intra0[rnd.next(0, i)]);
    for (int i = (int)intra1.size() - 1; i > 0; i--) swap(intra1[i], intra1[rnd.next(0, i)]);
    for (int i = (int)bridge.size() - 1; i > 0; i--) swap(bridge[i], bridge[rnd.next(0, i)]);

    int M = (int)intra0.size() + (int)intra1.size() + (int)bridge.size();

    printf("%d %d %.6f %.6f %d %d\n", N, M, R, C, T, W);
    for (int i = 1; i <= N; i++) printf("%.6f%c", x0[i], i == N ? '\n' : ' ');
    for (auto &e : intra0) printf("%d %d %.6f\n", e.first, e.second, INTRA_CAP);
    for (auto &e : intra1) printf("%d %d %.6f\n", e.first, e.second, INTRA_CAP);
    for (auto &e : bridge) printf("%d %d %.6f\n", e.first, e.second, BRIDGE_CAP);
    return 0;
}
