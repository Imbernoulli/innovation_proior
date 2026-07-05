#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- structure ladder: horizon grows from tiny (example scale) to large ----
    // testId 1 tiny (~8 slots), testId 10 large (~1808 slots).
    int T = 8 + (testId - 1) * 200;

    int q = 6 + (testId % 4);                 // on-demand throughput 6..9
    int gmax = 8 + (testId % 3) * 3;          // spot gain ceiling 8..14
    int K = 25 + (testId % 5) * 30;           // startup cost 25..145 (couples slots)

    // availability of the shared grid varies per test
    double availP = 0.50 + 0.03 * (testId % 5); // 0.50..0.62

    vector<int> avail(T + 1), g(T + 1), sc(T + 1), dc(T + 1);
    for (int t = 1; t <= T; t++) {
        avail[t] = (rnd.next(0.0, 1.0) < availP) ? 1 : 0;
        g[t] = rnd.next(1, gmax);
        // spot is cheap per sack (surplus power); on-demand pricier
        sc[t] = rnd.next(1, 12 + (testId % 3) * 6);   // 1..(12..24)
        dc[t] = rnd.next(24, 24 + 40 + (testId % 4) * 6); // pricier metered power
    }

    // demand: a fraction of what on-demand alone could mill, so it is always
    // satisfiable (q*T >= W) yet forces a real cost trade-off.
    double alpha = 0.40 + 0.04 * (testId % 4);        // 0.40..0.52
    long long W = (long long)floor(alpha * (double)q * (double)T);
    if (W < 1) W = 1;
    if (W > (long long)q * T) W = (long long)q * T;

    printf("%d %lld %d %d\n", T, W, q, K);
    for (int t = 1; t <= T; t++)
        printf("%d %d %d %d\n", avail[t], g[t], sc[t], dc[t]);
    return 0;
}
