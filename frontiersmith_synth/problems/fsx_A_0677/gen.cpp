#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// -----------------------------------------------------------------------------
// "Raked zen garden with rocks" (generator)  family: raked-garden-flowlines
//
// Grid R x C, some cells are rocks ('#'), rest is sand ('.'). A solution is a set
// of vertex-disjoint rake lines (grid paths, length >= 2) over the sand, whose
// total coverage must land in a permille band [CovMin, CovMax] of the sand cells.
// The objective (checker, minimized) is:
//   weighted-uncovered-cost (uncovered ring cells -- graph distance <= M to the
//   nearest rock -- cost Wr each, other uncovered sand costs 1 each)
//   + Sc * (population variance of each sand cell's graph-distance to the
//     nearest COVERED cell)
//
// TRAP: the coverage band is kept moderate (roughly 35-55%), so the "obvious"
// strategy -- rule a fixed number of evenly spaced FULL rows straight across the
// whole garden -- necessarily SKIPS a comparable fraction of rows. A rock's ring
// band spans 2M+1 rows; because the skip pattern is uniform and rock-agnostic,
// on average close to (1 - coverage fraction) of every ring's rows are skipped,
// regardless of where the rock sits -- a systematic miss, not a fluke. A
// level-set solver that unconditionally extracts {D(cell) <= M} first pays for
// this out of the SAME shared budget but never misses a ring cell. Clustered
// rocks (merged basins) and larger M on the bigger tests sharpen the effect.
// -----------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;

    int R = 10 + (int)llround(f * 60.0);   // 10 .. 70
    int C = 10 + (int)llround(f * 60.0);   // 10 .. 70

    int M = 1 + testId / 3;                // 1 .. 4  ring radius
    int Wr = 4 + (testId % 5);             // 4 .. 8  ring penalty weight
    int Sc = 6 + (testId % 4);             // 6 .. 9  spacing coefficient
    int covMin = 330 + 15 * (testId % 6);  // 330 .. 405  (~a third of the sand)
    int covMax = covMin + 160;             // ~16-percentage-point band width

    int K;                                   // number of rocks
    bool clustered;                          // pack some rocks into tight bunches
    switch (testId) {
        case 1: K = 1;  clustered = false; break;
        case 2: K = 2;  clustered = false; break;
        case 3: K = 3;  clustered = true;  break;   // merged basin
        case 4: K = 5;  clustered = false; break;
        case 5: K = 6;  clustered = true;  break;   // merged basin
        case 6: K = 7;  clustered = false; break;
        case 7: K = 9;  clustered = true;  break;   // merged basin
        case 8: K = 11; clustered = false; break;
        case 9: K = 15; clustered = true;  break;   // merged basin
        default: K = 20; clustered = true;  break;   // largest, adversarial mix
    }

    vector<string> g(R, string(C, '.'));

    // A handful of independent cluster nuclei (not one single point) so merged
    // basins appear in different parts of the grid, plus scattered singles.
    int nClusters = clustered ? max(1, K / 4) : 0;
    vector<pair<int,int>> anchors;
    for (int i = 0; i < nClusters; i++)
        anchors.push_back({rnd.next(0, R - 1), rnd.next(0, C - 1)});

    int placed = 0, attempts = 0;
    while (placed < K && attempts < K * 300 + 500) {
        attempts++;
        int rr, cc;
        if (clustered && !anchors.empty() && rnd.next(0, 2) > 0) {
            auto &a = anchors[rnd.next(0, (int)anchors.size() - 1)];
            rr = a.first + rnd.next(-M, M);
            cc = a.second + rnd.next(-M, M);
        } else {
            rr = rnd.next(0, R - 1);
            cc = rnd.next(0, C - 1);
        }
        rr = min(max(rr, 0), R - 1);
        cc = min(max(cc, 0), C - 1);
        if (g[rr][cc] == '#') continue;
        g[rr][cc] = '#';
        placed++;
    }

    printf("%d %d\n", R, C);
    printf("%d %d %d %d %d\n", M, Wr, covMin, covMax, Sc);
    for (auto &row : g) printf("%s\n", row.c_str());
    return 0;
}
