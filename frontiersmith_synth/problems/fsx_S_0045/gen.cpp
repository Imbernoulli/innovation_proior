#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- structure ladder: cold-chain interference graphs ----
    // testId 1 tiny (example scale); grows large by testId 10.
    // The graph is DENSE relative to the small channel budget C, so a conflict-free
    // assignment is impossible and the minimum interference is ~a 1/C fraction of B.
    int n = 5 + (testId - 1) * 25;          // 5, 30, 55, ..., 230
    int C = 2 + (testId % 3);               // 2, 3, 4 compressor circuits (small budget)

    // dense interference -> forced conflicts; density varies per test.
    double dens = 0.35 + (testId % 4) * 0.07; // 0.35 .. 0.56 fraction of all pairs
    // severity model: even tests heavy-tailed severities, odd tests mild.
    int Wmax = (testId % 2 == 0) ? 20 : 5;

    long long maxE = (long long)n * (n - 1) / 2;
    long long target = (long long)llround(dens * (double)maxE);
    if (target > maxE) target = maxE;
    if (target < 1) target = 1;

    // enumerate all candidate pairs, shuffle, take the first `target`.
    vector<pair<int,int>> all;
    all.reserve((size_t)maxE);
    for (int i = 1; i <= n; i++)
        for (int j = i + 1; j <= n; j++)
            all.push_back({i, j});
    shuffle(all.begin(), all.end());

    int m = (int)target;
    printf("%d %d %d\n", n, m, C);
    for (int e = 0; e < m; e++) {
        int u = all[e].first, v = all[e].second;
        int w;
        if (Wmax == 20 && rnd.next(0, 3) == 0)
            w = rnd.next(12, Wmax);         // occasional heavy severity
        else
            w = rnd.next(1, (Wmax == 20 ? 8 : Wmax));
        printf("%d %d %d\n", u, v, w);
    }
    return 0;
}
