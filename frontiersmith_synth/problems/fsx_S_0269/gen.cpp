#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Traffic-signal-grid job-shop (family scheduling-jssp), small scale, variant #12.
// testId is a difficulty ladder: tiny 2x2 grid / few convoys (testId 1, example scale)
// growing to a 3x4 grid with many contending convoys and heavy-tailed green windows.
int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int Rg = (testId <= 5) ? 2 : 3;             // grid rows
    int Cg = 2 + (testId - 1) / 3;              // grid cols: 2,2,2,3,3,3,4,4,4,5
    if (Cg > 4) Cg = 4;
    int M = Rg * Cg;                            // intersections (machines): 4..12

    int J = 3 + testId;                         // convoys (jobs): 4..13
    if (J > 15) J = 15;

    int cheapHi = 9;                            // typical green window ceiling
    int heavyLo = 12, heavyHi = 20;             // heavy-tailed bottleneck windows
    double heavyProb = 0.10 + 0.02 * (testId % 3);

    // upper bound on route length grows with the instance
    int Lcap = 3 + testId / 2;                  // 3..8
    if (Lcap > 8) Lcap = 8;
    if (Lcap > M) Lcap = M;

    printf("%d %d\n", M, J);
    for (int j = 0; j < J; j++) {
        int L = rnd.next(2, Lcap);
        printf("%d", L);
        int prev = -1;
        for (int i = 0; i < L; i++) {
            int m = rnd.next(1, M);
            while (m == prev) m = rnd.next(1, M);   // no immediate repeat of the same light
            prev = m;
            int p;
            if (rnd.next(0.0, 1.0) < heavyProb)
                p = rnd.next(heavyLo, heavyHi);
            else
                p = rnd.next(1, cheapHi);
            printf(" %d %d", m, p);
        }
        printf("\n");
    }
    return 0;
}
