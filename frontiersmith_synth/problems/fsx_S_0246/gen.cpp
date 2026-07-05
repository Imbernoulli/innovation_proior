#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Curator's Conundrum: generalized assignment (multiple-knapsack) generator.
// testId is a difficulty/structure ladder:
//   testId 1  -> tiny (example scale), few groups/galleries, roomy budgets
//   testId 10 -> large & tight, heavy-tailed profits, adversarial structure
int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int V = 12 + 22 * (testId - 1);   // 12, 34, ..., 210
    int G = 2 + (testId + 1) / 2;     // 2, 3, 3, 4, 4, 5, 5, 6, 6, 7  -> small
    if (V > 220) V = 220;
    if (G > 16) G = 16;

    // cost model: docent-minutes per (group, gallery)
    int aHi = 4 + (testId % 5);       // 4..8 ceiling on per-tour cost

    // per-gallery budgets: tight enough that only a handful of tours fit.
    // C_j >= aHi guarantees every single group fits any gallery alone.
    vector<int> C(G + 1);
    for (int j = 1; j <= G; j++) {
        // tight budgets: each gallery holds only a couple of tours, so total value
        // stays a few-fold above the single-best baseline (ratios do NOT saturate).
        int base = aHi + rnd.next(1, aHi + 2);
        if (base > 60) base = 60;
        C[j] = base;
    }

    // profit model: mostly modest, with a moderate tail so the single-best baseline
    // is sizeable but total achievable value stays a few-fold above it.
    // This keeps ratios spread across [0.1, ~1.0] rather than all saturating.
    double tailProb = 0.10 + 0.01 * (testId % 4); // fraction of high-value tours

    vector<vector<int>> a(V + 1, vector<int>(G + 1));
    vector<vector<int>> p(V + 1, vector<int>(G + 1));
    for (int i = 1; i <= V; i++) {
        for (int j = 1; j <= G; j++) {
            int cost = rnd.next(1, aHi);
            if (cost > C[j]) cost = C[j];
            a[i][j] = cost;
            int prof;
            if (rnd.next(0.0, 1.0) < tailProb)
                prof = rnd.next(250, 500);    // moderate tail
            else
                prof = rnd.next(1, 90);       // modest bulk
            p[i][j] = prof;
        }
    }

    // print
    printf("%d %d\n", V, G);
    for (int j = 1; j <= G; j++) {
        printf("%d%c", C[j], j == G ? '\n' : ' ');
    }
    for (int i = 1; i <= V; i++) {
        for (int j = 1; j <= G; j++) {
            printf("%d %d%c", a[i][j], p[i][j], j == G ? '\n' : ' ');
        }
    }
    return 0;
}
