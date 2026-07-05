#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    int N, T, K, D;
    double pav;      // availability density
    int spotLo, spotHi;
    int lamLo, lamHi;

    if (t <= 1) {
        // tiny, example-scale sanity instance
        N = 2; T = 4; K = 1; D = 100;
        pav = 0.6; spotLo = 20; spotHi = 250; lamLo = 80; lamHi = 300;
    } else {
        N = 8 + t * 12;                 // up to ~128
        T = 20 + t * 17;                // up to ~190
        K = max(1, N / (3 + (t % 3)));  // scarce shared crew
        D = 100;
        // odd tests: sparse volunteers (rangers matter); even: denser
        pav = (t % 2 == 1) ? (0.55 - 0.02 * t) : (0.75 - 0.02 * t);
        if (pav < 0.2) pav = 0.2;
        // volatile spot market straddling D and lambda
        spotLo = 10; spotHi = 260;
        lamLo = 80;  lamHi = 300;
    }
    if (K > N) K = N;

    printf("%d %d %d %d\n", N, T, K, D);

    // spot market price per step
    for (int j = 0; j < T; j++) {
        int s = rnd.next(spotLo, spotHi);
        printf("%d%c", s, j + 1 == T ? '\n' : ' ');
    }

    // required surveillance units per tower: high enough that spot alone
    // often cannot cover, forcing crew contention / penalty tradeoffs
    vector<int> R(N);
    for (int i = 0; i < N; i++) {
        int lo = max(1, (int)(T * 0.35));
        int hi = max(lo, (int)(T * 0.85));
        R[i] = rnd.next(lo, hi);
        printf("%d%c", R[i], i + 1 == N ? '\n' : ' ');
    }

    // per-unit liability, straddling D so ranger-vs-penalty varies per tower
    for (int i = 0; i < N; i++) {
        int lam = rnd.next(lamLo, lamHi);
        printf("%d%c", lam, i + 1 == N ? '\n' : ' ');
    }

    // availability matrix
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < T; j++) {
            int a = (rnd.next(0, 999) < (int)(pav * 1000)) ? 1 : 0;
            printf("%d%c", a, j + 1 == T ? '\n' : ' ');
        }
    }
    return 0;
}
