#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Job-shop scheduling with recirculation, themed "quantum lab wiring".
// testId is a difficulty ladder: testId 1 is tiny (example scale),
// growing to a large, dense multi-machine shop by testId 10.
int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int M = 2 + testId;                 // 3 .. 12 workstations
    int J = 3 * testId;                 // 3 .. 30 modules
    if (J > 40) J = 40;

    // operations per module ~ M, with mild variation -> recirculation and
    // non-uniform routing (some machines revisited, some skipped).
    int nopsBase = M;

    // duration model varies per test: some uniform, some skewed toward a
    // few "slow" workstations, which changes where the bottleneck is.
    int durHi = 20 + (testId % 5) * 15;         // 20 .. 80 nominal ceiling
    if (durHi > 99) durHi = 99;
    double heavyProb = 0.10 + 0.03 * (testId % 4); // fraction of slow ops
    int heavyLo = min(99, durHi + 5);
    int heavyHi = 99;

    // choose which workstations are "slow" this test (skew)
    int slowCount = 1 + (testId % 3);
    set<int> slow;
    while ((int)slow.size() < min(slowCount, M)) slow.insert(rnd.next(1, M));

    // build jobs, respecting the total-operation cap
    int cap = 500;
    vector<vector<pair<int,int>>> jobs;   // (machine, dur)
    int total = 0;
    for (int j = 0; j < J && total < cap; j++) {
        int k = nopsBase + rnd.next(-1, 2);       // M-1 .. M+2 operations
        if (k < 1) k = 1;
        if (total + k > cap) k = cap - total;
        vector<pair<int,int>> ops;
        for (int o = 0; o < k; o++) {
            int m = rnd.next(1, M);
            int d;
            bool isSlow = slow.count(m);
            if (isSlow || rnd.next(0.0, 1.0) < heavyProb)
                d = rnd.next(heavyLo, heavyHi);
            else
                d = rnd.next(1, durHi);
            ops.push_back({m, d});
        }
        jobs.push_back(ops);
        total += k;
    }

    J = (int)jobs.size();
    printf("%d %d\n", J, M);
    for (auto& ops : jobs) {
        printf("%d", (int)ops.size());
        for (auto& p : ops) printf(" %d %d", p.first, p.second);
        printf("\n");
    }
    return 0;
}
