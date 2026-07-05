#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- structure ladder: downtown traffic-signal grid ----
    // testId 1 is tiny (example scale); grows to a dense job board by testId 10.
    int G = 8 + 2 * testId;              // coordinate span 10..28
    int N = 6 + 3 * testId;              // 9..36 intersections
    int M = 4 + 3 * testId;              // 7..34 jobs
    if (N > 40) N = 40;
    if (M > 40) M = 40;

    // vary the metric per test: some grids have costly signals, some cheap;
    // penalty richness (how profitable jobs are) also varies -> prize-collecting
    // optimum shifts between "serve few" and "serve many".
    int delayHi = testId % 4;                       // 0..3 max red-light wait
    double alphaLo = 0.8 + 0.05 * (testId % 5);     // penalty scale floor
    double alphaSpread = 0.6 + 0.1 * (testId % 4);  // extra multiplicative noise

    // ---- intersections ----
    vector<int> X(N + 1), Y(N + 1), D(N + 1);
    for (int i = 1; i <= N; i++) {
        X[i] = rnd.next(0, G);
        Y[i] = rnd.next(0, G);
        D[i] = (delayHi == 0) ? 0 : rnd.next(0, delayHi);
    }
    // put the depot near the centre of the grid
    int depot = 1;
    X[depot] = G / 2;
    Y[depot] = G / 2;
    D[depot] = 0;

    auto man = [&](int a, int b) {
        return abs(X[a] - X[b]) + abs(Y[a] - Y[b]);
    };
    // arrival cost a -> b (matches statement dist(a,b))
    auto darr = [&](int a, int b) {
        return man(a, b) + D[b];
    };

    // ---- jobs ----
    struct Job { int p, q, w; };
    vector<Job> jobs;
    for (int j = 0; j < M; j++) {
        int p = rnd.next(1, N);
        int q = rnd.next(1, N);
        while (q == p) q = rnd.next(1, N);
        // direct courier cost of this job in isolation
        int direct = darr(depot, p) + darr(p, q) + darr(q, depot);
        double alpha = alphaLo + rnd.next(0.0, alphaSpread);
        int w = (int)llround(alpha * (double)direct) + rnd.next(1, 20);
        if (w < 1) w = 1;
        if (w > 5000) w = 5000;
        jobs.push_back({p, q, w});
    }

    printf("%d %d %d\n", N, M, depot);
    for (int i = 1; i <= N; i++)
        printf("%d %d %d\n", X[i], Y[i], D[i]);
    for (auto& jb : jobs)
        printf("%d %d %d\n", jb.p, jb.q, jb.w);
    return 0;
}
