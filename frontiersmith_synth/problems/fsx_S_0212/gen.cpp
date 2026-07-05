#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Planted balanced maximum-cut ("AGV fleet zoning").
// A hidden balanced ground-truth partition carries the HEAVY cross-fleet weight;
// lighter intra-fleet and noise edges add tension so the alternating index baseline
// (the checker's B) is uncorrelated with the good cut. testId is the difficulty ladder.

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int n = 24 * testId;             // 24, 48, ..., 240 (even)
    int D = testId % 3;              // fleet-size tolerance 0, 1, 2

    // hidden balanced labeling: random assignment of exactly n/2 to each side
    vector<int> perm(n);
    for (int i = 0; i < n; i++) perm[i] = i + 1;
    shuffle(perm.begin(), perm.end());       // testlib deterministic shuffle
    vector<int> lab(n + 1, 0);
    for (int i = 0; i < n; i++) lab[perm[i]] = (i < n / 2) ? 0 : 1;

    vector<int> side0, side1;
    for (int i = 1; i <= n; i++) (lab[i] == 0 ? side0 : side1).push_back(i);

    int hlo = 8 + (testId % 4);      // heavy cross-fleet congestion floor 8..11
    int hhi = 20 + (testId % 6);     // heavy ceiling 20..25
    int llo = 1, lhi = 5;            // light intra / noise congestion

    vector<array<int, 3>> edges;

    // heavy cross-fleet conflicts (the planted signal)
    int crossCnt = 3 * n;
    for (int e = 0; e < crossCnt; e++) {
        int u = side0[rnd.next(0, (int)side0.size() - 1)];
        int v = side1[rnd.next(0, (int)side1.size() - 1)];
        int w = rnd.next(hlo, hhi);
        edges.push_back({u, v, w});
    }

    // lighter intra-fleet conflicts (tempt the solver to break the planted cut)
    int intraCnt = n;
    for (int e = 0; e < intraCnt; e++) {
        vector<int>& S = (rnd.next(0, 1) == 0) ? side0 : side1;
        int u = S[rnd.next(0, (int)S.size() - 1)];
        int v = S[rnd.next(0, (int)S.size() - 1)];
        if (u == v) { e--; continue; }
        int w = rnd.next(llo, lhi);
        edges.push_back({u, v, w});
    }

    // light noise conflicts between arbitrary cells
    int noiseCnt = n;
    for (int e = 0; e < noiseCnt; e++) {
        int u = rnd.next(1, n), v = rnd.next(1, n);
        if (u == v) { e--; continue; }
        int w = rnd.next(llo, lhi);
        edges.push_back({u, v, w});
    }

    shuffle(edges.begin(), edges.end());     // decouple edge index from structure
    int m = (int)edges.size();
    printf("%d %d %d\n", n, m, D);
    for (auto& e : edges) printf("%d %d %d\n", e[0], e[1], e[2]);
    return 0;
}
