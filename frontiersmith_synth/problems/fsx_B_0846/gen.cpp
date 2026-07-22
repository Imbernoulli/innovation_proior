#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// -----------------------------------------------------------------------------
// "Pick a team that clicks but does not crowd"   family: synergy-congestion-team
//
// We build a graph out of three kinds of blocks:
//   - ISOLATED filler nodes: zero edges. Always safe to pick; their value sum
//     is exactly the checker's baseline B.
//   - TRAP CLIQUES: a fully-connected cluster with per-node crowd cap `cap`
//     (small) and crowd penalty rate `pen` (large relative to the edge
//     synergy `s`). Selecting the WHOLE clique overloads everyone's crowd
//     cap; the true optimum keeps only an interior core of size ~cap+1.
//     A myopic marginal-gain greedy that (when scoring candidate k) only
//     charges k's OWN resulting crowd penalty -- and never re-prices the
//     crowd charge it retroactively imposes on already-selected neighbours
//     of k -- keeps seeing a "positive" reason to add more clique members
//     long after the true objective has turned negative. This is the planted
//     trap: greedy walks straight into an overcrowded clique.
//   - NOISE blocks: a sparser random graph where every node's crowd cap is
//     set to (at least) its own degree in that block, so taking ANY subset
//     of it never triggers a penalty -- free, safe extra value that rewards
//     simply being thorough, to keep the instance from being ONLY about the
//     trap.
//
// Node ids are finally passed through a random permutation so a solver can't
// shortcut by assuming block order (must actually read v/cap/pen/edges).
// -----------------------------------------------------------------------------

static vector<int> V, CAP, PEN;
static vector<array<int,3>> E; // i, j, s  (temp ids, i<j)

static int addIsolated(int count, int vlo, int vhi) {
    int start = (int)V.size();
    for (int t = 0; t < count; t++) {
        V.push_back(rnd.next(vlo, vhi));
        CAP.push_back(0);
        PEN.push_back(0);
    }
    return start;
}

static int addClique(int size, int vlo, int vhi, int slo, int shi, int cap, int pen) {
    int start = (int)V.size();
    for (int t = 0; t < size; t++) {
        V.push_back(rnd.next(vlo, vhi));
        CAP.push_back(cap);
        PEN.push_back(pen);
    }
    for (int a = 0; a < size; a++)
        for (int b = a + 1; b < size; b++)
            E.push_back({start + a, start + b, rnd.next(slo, shi)});
    return start;
}

static int addNoiseSafe(int count, double avgDeg, int vlo, int vhi, int slo, int shi) {
    int start = (int)V.size();
    for (int t = 0; t < count; t++) {
        V.push_back(rnd.next(vlo, vhi));
        CAP.push_back(0);   // fixed up below
        PEN.push_back(rnd.next(3, 20));
    }
    if (count >= 2) {
        int target = max(0, (int)llround(avgDeg * count / 2.0));
        set<pair<int,int>> used;
        vector<int> deg(count, 0);
        int guard = 0;
        while ((int)used.size() < target && guard++ < target * 20 + 200) {
            int a = rnd.next(0, count - 1), b = rnd.next(0, count - 1);
            if (a == b) continue;
            if (a > b) swap(a, b);
            if (!used.insert({a, b}).second) continue;
            E.push_back({start + a, start + b, rnd.next(slo, shi)});
            deg[a]++; deg[b]++;
        }
        for (int t = 0; t < count; t++) CAP[start + t] = deg[t]; // never crowds
    }
    return start;
}

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    if (testId == 1) {
        // Fixed tiny worked example (matches statement.txt exactly).
        V = {30,30,30,30,50,60};
        CAP = {2,2,2,2,0,0};
        PEN = {20,20,20,20,0,0};
        E = {{0,1,10},{0,2,10},{0,3,10},{1,2,10},{1,3,10},{2,3,10}};
    } else {
        int idx = testId - 2; // 0..8
        // Trap cliques are kept a LARGE, non-diluting share of each instance
        // (filler/noise stay modest) so the greedy-vs-strong gap the traps
        // create does not get washed out by baseline size as N grows.
        static const int nTrap[9]    = {1, 2, 2, 1, 3, 4, 6, 9,13};
        static const int cliqLo[9]   = {6, 7, 8, 5, 7, 8, 9,11,13};
        static const int cliqHi[9]   = {9,11,13, 7,11,14,17,20,26};
        static const int cliqCap[9]  = {3, 3, 3, 2, 3, 3, 2, 3, 3};
        static const int noiseN[9]   = {5, 8,10,20,10,15,20,28,38};
        static const int fillerN[9]  = {4, 5, 6, 5, 6, 7, 8, 9,11};

        int nc = nTrap[idx];
        for (int c = 0; c < nc; c++) {
            int sz = rnd.next(cliqLo[idx], cliqHi[idx]);
            int cap = cliqCap[idx];
            int slo = 8, shi = 20;
            int pen = rnd.next(34, 52); // comfortably > shi -> real crowd trap
            addClique(sz, 30, 75, slo, shi, cap, pen);
        }
        addNoiseSafe(noiseN[idx], 3.0, 25, 90, 6, 18);
        addIsolated(fillerN[idx], 60, 190);
    }

    int N = (int)V.size();
    // random relabeling so block order isn't a solvable shortcut
    vector<int> perm = rnd.perm(N);
    vector<int> v2(N), cap2(N), pen2(N);
    for (int i = 0; i < N; i++) { v2[perm[i]] = V[i]; cap2[perm[i]] = CAP[i]; pen2[perm[i]] = PEN[i]; }
    vector<array<int,3>> e2;
    e2.reserve(E.size());
    for (auto& t : E) {
        int a = perm[t[0]], b = perm[t[1]];
        if (a > b) swap(a, b);
        e2.push_back({a, b, t[2]});
    }
    sort(e2.begin(), e2.end(), [](const array<int,3>& x, const array<int,3>& y) {
        if (x[0] != y[0]) return x[0] < y[0];
        return x[1] < y[1];
    });

    printf("%d %d\n", N, (int)e2.size());
    for (int i = 0; i < N; i++) printf("%d%c", v2[i], " \n"[i + 1 == N]);
    for (int i = 0; i < N; i++) printf("%d%c", cap2[i], " \n"[i + 1 == N]);
    for (int i = 0; i < N; i++) printf("%d%c", pen2[i], " \n"[i + 1 == N]);
    for (auto& t : e2) printf("%d %d %d\n", t[0], t[1], t[2]);
    return 0;
}
