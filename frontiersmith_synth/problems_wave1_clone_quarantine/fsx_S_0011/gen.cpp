#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// gen.cpp -- satellite ground-station activation (max-weight independent set on a
// general conflict graph).  testId is a difficulty/structure ladder: tiny example-scale
// at 1, growing to larger + denser + weight-skewed instances toward 10.  Denser graphs
// shrink the achievable independent-set value relative to the single-best station, which
// prevents the score from trivially saturating and separates heuristics.
int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int N;
    int avgdeg;
    bool skewed;

    if (testId == 1) {
        // tiny example-scale sanity instance
        N = 8; avgdeg = 3; skewed = false;
    } else {
        N = min(250, 30 + (testId - 1) * 25);   // 55, 80, ..., 250 (capped)
        avgdeg = 3 + testId;                     // 5,6,7,...,13  (density grows)
        skewed = (testId % 2 == 1);              // odd tests: heavy-tailed values
    }

    long long targetM = (long long)N * avgdeg / 2;
    if (targetM < 1) targetM = 1;
    if (targetM > 6000) targetM = 6000;
    // cannot have more distinct pairs than C(N,2); allow duplicates but keep sane
    long long maxPairs = (long long)N * (N - 1) / 2;
    if (targetM > 2 * maxPairs) targetM = maxPairs;
    int M = (int)targetM;

    // weights
    vector<int> w(N + 1);
    for (int j = 1; j <= N; j++) {
        if (skewed) {
            // mostly modest stations with a few very high-value ones
            if (rnd.next(0, 11) == 0) w[j] = rnd.next(700, 1000);
            else w[j] = rnd.next(1, 120);
        } else {
            w[j] = rnd.next(1, 1000);
        }
    }

    // edges (general graph); duplicates allowed and describe the same interference
    vector<pair<int,int>> edges;
    edges.reserve(M);
    for (int i = 0; i < M; i++) {
        int u = rnd.next(1, N);
        int v = rnd.next(1, N);
        while (v == u) v = rnd.next(1, N);
        edges.push_back({u, v});
    }

    printf("%d %d\n", N, M);
    for (int j = 1; j <= N; j++)
        printf("%d%c", w[j], j == N ? '\n' : ' ');
    for (auto &e : edges)
        printf("%d %d\n", e.first, e.second);
    return 0;
}
