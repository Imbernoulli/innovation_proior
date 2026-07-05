#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- structure ladder: warehouse job-shop instance ----
    // testId 1 is tiny (example scale); grows to a large heavy-tailed
    // instance by testId 10. M kept small (<=6) so the makespan/serial
    // ratio stays well below the 10x cap and dispatch heuristics diverge.
    int M = min(6, 2 + testId);                 // 3,4,5,6,6,6,...
    long long jj = 3 + (long long)(testId - 1) * (testId - 1) * 8;
    int J = (int)jj;                            // 3, 11, 35, 75, ..., 651

    // heavy-tailed durations: a few long "jam" operations create idle
    // gaps that separate good schedulers from bad ones.
    double heavyProb = 0.10 + 0.01 * (testId % 5); // 0.10..0.14

    printf("%d %d\n", J, M);

    vector<int> perm(M);
    for (int m = 0; m < M; m++) perm[m] = m + 1;

    for (int j = 0; j < J; j++) {
        shuffle(perm.begin(), perm.end());      // random bay route per order
        for (int o = 0; o < M; o++) {
            int d;
            if (rnd.next(0.0, 1.0) < heavyProb)
                d = rnd.next(200, 1000);        // long jam step
            else
                d = rnd.next(1, 60);            // ordinary step
            printf("%d %d", perm[o], d);
            if (o + 1 < M) printf(" ");
        }
        printf("\n");
    }
    return 0;
}
