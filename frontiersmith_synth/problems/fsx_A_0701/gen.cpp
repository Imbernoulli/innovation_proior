// Generator for fsx_A_0701 "Fold a Big Network onto a Small Labeled One"
#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // Ladder of (k0, n0, corruption fraction, extra chord count), testId 1..10.
    static const int    K0[11]      = {0, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8};
    static const int    N0[11]      = {0, 5, 8, 15, 30, 40, 80, 100, 150, 175, 250};
    static const double CORR[11]    = {0, 0.05, 0.06, 0.08, 0.10, 0.10, 0.12, 0.10, 0.13, 0.12, 0.15};
    static const int    CHORDS[11]  = {0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4};

    int k0 = K0[testId], n0 = N0[testId], chords = CHORDS[testId];
    double corr = CORR[testId];
    int N = k0 * n0;

    // 1. Build the base multigraph: a k0-cycle with voltage 1 (guarantees the
    //    unrewired lift is connected), plus `chords` random extra edges.
    vector<array<int,3>> baseEdges; // (u,v,g)
    for (int i = 0; i < k0; i++) {
        baseEdges.push_back({i, (i + 1) % k0, 1});
    }
    for (int c = 0; c < chords; c++) {
        int u = rnd.next(k0), v = rnd.next(k0);
        while (v == u) v = rnd.next(k0);
        int g = rnd.next(0, n0 - 1);
        baseEdges.push_back({u, v, g});
    }

    // 2. Lift: vertex id VID(u,i) = u*n0+i, 0-indexed, for u in [0,k0), i in [0,n0).
    auto VID = [&](int u, int i) { return u * n0 + i; };
    set<pair<int,int>> edgeSet;
    for (auto& e : baseEdges) {
        int u = e[0], v = e[1], g = e[2];
        for (int i = 0; i < n0; i++) {
            int a = VID(u, i);
            int b = VID(v, (i + g) % n0);
            if (a == b) continue;
            int lo = min(a, b), hi = max(a, b);
            edgeSet.insert({lo, hi});
        }
    }

    // 3. Corruption: remove a random fraction of edges, add the same number of
    //    fresh random edges (avoiding self-loops and duplicates).
    vector<pair<int,int>> edgeVec(edgeSet.begin(), edgeSet.end());
    shuffle(edgeVec.begin(), edgeVec.end());
    int M0 = (int)edgeVec.size();
    int numCorrupt = (int)llround(corr * M0);
    numCorrupt = min(numCorrupt, M0 - 1); // never wipe out everything
    set<pair<int,int>> finalEdges(edgeVec.begin() + numCorrupt, edgeVec.end());
    int added = 0;
    int guard = 0;
    while (added < numCorrupt && guard < 200000) {
        guard++;
        int a = rnd.next(0, N - 1), b = rnd.next(0, N - 1);
        if (a == b) continue;
        int lo = min(a, b), hi = max(a, b);
        if (finalEdges.count({lo, hi})) continue;
        finalEdges.insert({lo, hi});
        added++;
    }

    // 4. Relabel: random permutation of vertex ids -> external 1-indexed ids.
    vector<int> perm = rnd.perm(N, 0); // perm[internal] = external (0-indexed)
    vector<pair<int,int>> outEdges;
    outEdges.reserve(finalEdges.size());
    for (auto& e : finalEdges) {
        int a = perm[e.first] + 1, b = perm[e.second] + 1;
        outEdges.push_back({min(a,b), max(a,b)});
    }
    shuffle(outEdges.begin(), outEdges.end());

    int M = (int)outEdges.size();
    printf("%d %d\n", N, M);
    for (auto& e : outEdges) {
        printf("%d %d\n", e.first, e.second);
    }
    return 0;
}
