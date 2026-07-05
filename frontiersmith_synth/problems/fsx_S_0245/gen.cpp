#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    // Difficulty ladder: tiny at t=1, medium/dense at t=10.
    int J = 2 + (t - 1) * 3;          // 2 .. 29 inverters
    if (J < 2) J = 2;
    if (J > 30) J = 30;
    int M = 2 + (t - 1);              // 2 .. 11 test bays
    if (M < 2) M = 2;
    if (M > 15) M = 15;

    // Structure variety by testId: some tests uniform durations, some skewed
    // (a few heavy stages), some with correlated bay bottlenecks.
    int mode = t % 3;                 // 0 uniform, 1 skewed, 2 bottleneck

    printf("%d %d\n", J, M);
    for (int i = 0; i < J; i++) {
        // Each inverter visits every bay exactly once, in a random order.
        vector<int> perm(M);
        for (int m = 0; m < M; m++) perm[m] = m + 1;
        shuffle(perm.begin(), perm.end());

        printf("%d", M);
        for (int k = 0; k < M; k++) {
            int bay = perm[k];
            int d;
            if (mode == 0) {
                d = rnd.next(1, 100);
            } else if (mode == 1) {
                // skewed: mostly short, occasional heavy stage
                if (rnd.next(0, 4) == 0) d = rnd.next(60, 100);
                else d = rnd.next(1, 25);
            } else {
                // bottleneck: one bay tends to carry long stages
                int hot = 1 + (i % M);   // deterministic-ish hot bay rotation
                if (bay == hot) d = rnd.next(50, 100);
                else d = rnd.next(1, 40);
            }
            printf(" %d %d", bay, d);
        }
        printf("\n");
    }
    return 0;
}
