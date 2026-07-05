#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// gen.cpp -- satellite ground-station channel separation (weighted T-coloring / FAP).
// testId is a difficulty/structure ladder: tiny sparse at 1 -> large dense/skewed/adversarial at 10.
// Deterministic given testId (uses testlib rnd only).
int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int N, K, avgdeg;
    bool skewed, clustered;

    if (testId == 1) {
        // tiny, example-scale sanity instance
        N = 6; K = 3; avgdeg = 3; skewed = false; clustered = false;
    } else {
        // grow N; keep K modest so separation demands are genuinely forced
        N = min(800, 30 + (testId - 1) * 80);        // 110,190,...,750 (capped 800)
        int kcycle[6] = {2, 3, 4, 3, 5, 4};
        K = kcycle[testId % 6];
        if (K < 2) K = 2;
        avgdeg = 5 + (testId % 5) * 3;                // 5..17
        if (testId == 10) avgdeg += 4;               // last test: densest / hardest
        skewed = (testId % 2 == 1);                  // odd tests: heavy-tailed weights
        clustered = (testId % 3 == 0);               // some tests: near-clique hot spots
    }

    long long targetM = (long long)N * avgdeg / 2;
    if (targetM < 1) targetM = 1;
    if (targetM > 8000) targetM = 8000;
    int M = (int)targetM;

    // Optional dense cluster: a subset of stations gets extra mutual interference,
    // creating a hard "hot spot" that forces residual cost with a small budget.
    int clusterSize = clustered ? min(N, max(4, K + 3)) : 0;

    vector<array<int,4>> edges;
    edges.reserve(M);

    auto pickW = [&](void) -> int {
        if (skewed) {
            if (rnd.next(0, 9) == 0) return rnd.next(500, 1000); // rare heavy coupling
            return rnd.next(1, 40);
        }
        return rnd.next(1, 100);
    };
    auto pickG = [&](void) -> int {
        // required separation in [1, K-1]; bias a little toward larger gaps when possible
        int hi = K - 1;
        if (hi <= 1) return 1;
        int g = rnd.next(1, hi);
        if (rnd.next(0, 3) == 0) g = rnd.next(1, hi);        // reshuffle -> mild bias to mid
        return g;
    };

    // Seed the cluster with dense pairs first (if any budget), then fill randomly.
    int placed = 0;
    if (clusterSize >= 2) {
        for (int a = 1; a <= clusterSize && placed < M; a++)
            for (int b = a + 1; b <= clusterSize && placed < M; b++) {
                edges.push_back({a, b, pickG(), pickW()});
                placed++;
            }
    }
    while (placed < M) {
        int u = rnd.next(1, N);
        int v = rnd.next(1, N);
        while (v == u) v = rnd.next(1, N);
        edges.push_back({u, v, pickG(), pickW()});
        placed++;
    }

    printf("%d %d %d\n", N, M, K);
    for (auto &e : edges)
        printf("%d %d %d %d\n", e[0], e[1], e[2], e[3]);
    return 0;
}
