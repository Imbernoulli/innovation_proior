#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    // testId ladder: N grows large, M grows modestly (keeps ratios below the 1.0 cap
    // so greedy vs strong stay distinguishable). t=1 tiny -> t=10 large/dense.
    int N = 3 + (t - 1) * 13;       // 3 .. 120
    if (N > 120) N = 120;
    int M = 3 + (t - 1) / 2;        // 3 .. 7
    if (M < 3) M = 3;
    if (M > 8) M = 8;

    // Duration regime varies with testId: some tests uniform, some skewed (a few long
    // stages) so distinct heuristics diverge.
    bool skewed = (t % 3 == 0);

    printf("%d %d\n", N, M);
    for (int i = 0; i < N; i++) {
        // random machine permutation for this loop
        vector<int> perm(M);
        for (int j = 0; j < M; j++) perm[j] = j;
        shuffle(perm.begin(), perm.end());
        for (int j = 0; j < M; j++) {
            int dur;
            if (skewed && rnd.next(0, 4) == 0) {
                dur = rnd.next(60, 99);      // occasional long bottleneck stage
            } else {
                dur = rnd.next(1, 40);
            }
            if (j) printf(" ");
            printf("%d %d", perm[j], dur);
        }
        printf("\n");
    }
    return 0;
}
