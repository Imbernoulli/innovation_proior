#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- structure ladder ----
    // testId 1 tiny (example scale), growing to large/adversarial by testId 10.
    int N = 4 + (testId - 1) * 6;           // 4 .. 58
    if (N > 60) N = 60;
    int T = 12 + (testId - 1) * 32;         // 12 .. 300
    if (T > 300) T = 300;

    int warm = rnd.next(20, 60);

    bool tight = (testId % 2 == 0);         // even tests: tight deadlines + scarce spot
    bool heavyTail = (testId >= 6);         // later tests: heavy-tailed spot spikes

    // prices
    vector<int> od(T + 1), sp(T + 1), C(T + 1);
    for (int t = 1; t <= T; t++) {
        od[t] = rnd.next(80, 120);
        int base;
        if (heavyTail && rnd.next(0.0, 1.0) < 0.30)
            base = rnd.next(40, 70);        // expensive spot spike
        else
            base = rnd.next(8, 28);         // cheap spot window
        sp[t] = base;

        // spot capacity: sometimes zero (spot unavailable), otherwise limited
        double zeroProb = tight ? 0.22 : 0.12;
        if (rnd.next(0.0, 1.0) < zeroProb) {
            C[t] = 0;
        } else {
            int capHi = max(1, N / (tight ? 4 : 2));
            C[t] = rnd.next(1, capHi);
        }
    }

    // wells: (R_j, d_j) with 1 <= R_j <= d_j <= T
    vector<pair<int,int>> wells(N);
    for (int j = 0; j < N; j++) {
        int R = rnd.next(1, max(1, T / 3));
        int d;
        if (tight) {
            // deadline close to R (little slack)
            int slack = rnd.next(0, max(0, T / 10));
            d = min(T, R + slack);
        } else {
            d = rnd.next(R, T);
        }
        if (d < R) d = R;
        wells[j] = {R, d};
    }

    // ---- print ----
    printf("%d %d %d\n", N, T, warm);
    for (int t = 1; t <= T; t++) printf("%d%c", od[t], t == T ? '\n' : ' ');
    for (int t = 1; t <= T; t++) printf("%d%c", sp[t], t == T ? '\n' : ' ');
    for (int t = 1; t <= T; t++) printf("%d%c", C[t], t == T ? '\n' : ' ');
    for (int j = 0; j < N; j++) printf("%d %d\n", wells[j].first, wells[j].second);

    return 0;
}
