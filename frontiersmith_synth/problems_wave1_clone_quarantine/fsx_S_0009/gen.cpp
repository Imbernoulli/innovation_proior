#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// gen.cpp -- greenhouse climate-program assignment (weighted min-conflict K-coloring)
// testId is a difficulty/structure ladder: tiny sparse at 1 -> large dense/skewed at 10.
int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int N, K, avgdeg;
    bool skewed;

    if (testId == 1) {
        // tiny, example-scale sanity instance
        N = 6; K = 2; avgdeg = 3; skewed = false;
    } else {
        // grow N; keep K small so conflicts are genuinely forced (degree > K)
        N = min(300, 20 + (testId - 1) * 30);   // 50,80,...,290 (capped 300)
        int kcycle[4] = {2, 3, 2, 4};
        K = kcycle[testId % 4];                  // small budget -> hard
        avgdeg = 4 + (testId % 5) * 2;           // 4..12
        skewed = (testId % 2 == 1);              // odd tests: heavy-tailed weights
    }

    long long targetM = (long long)N * avgdeg / 2;
    if (targetM < 1) targetM = 1;
    if (targetM > 5000) targetM = 5000;
    int M = (int)targetM;

    vector<array<int,3>> edges;
    edges.reserve(M);
    for (int i = 0; i < M; i++) {
        int u = rnd.next(1, N);
        int v = rnd.next(1, N);
        while (v == u) v = rnd.next(1, N);
        int w;
        if (skewed) {
            // mostly light couplings with a few very heavy ones
            if (rnd.next(0, 9) == 0) w = rnd.next(500, 1000);
            else w = rnd.next(1, 40);
        } else {
            w = rnd.next(1, 100);
        }
        edges.push_back({u, v, w});
    }

    printf("%d %d %d\n", N, M, K);
    for (auto &e : edges)
        printf("%d %d %d\n", e[0], e[1], e[2]);
    return 0;
}
