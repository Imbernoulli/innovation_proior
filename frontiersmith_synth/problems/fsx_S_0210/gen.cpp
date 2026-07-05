#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- scale ladder: small (example scale) at testId 1, medium/dense by testId 10 ----
    int P, B;
    if (testId == 1) { P = 2; B = 3; }
    else {
        P = 2 + 3 * (testId - 1);        // 5, 8, ..., 29
        if (P > 30) P = 30;
        B = 4 + 74 * (testId - 1);       // 78, 152, ..., 670
        if (B > 700) B = 700;
    }

    // per-block preferred pump creates "specialist" structure so that the
    // first-fit reference (pump-order biased) is far from optimal.
    vector<int> pref(B + 1);
    for (int j = 1; j <= B; j++) pref[j] = rnd.next(1, P);

    // weight / value matrices (pump-specific)
    vector<vector<int>> w(P + 1, vector<int>(B + 1));
    vector<vector<int>> v(P + 1, vector<int>(B + 1));

    // skew of the value distribution varies per test
    int vHi = 120 + (testId % 4) * 40;   // 120..240
    for (int j = 1; j <= B; j++) {
        int base = rnd.next(20, vHi);    // block intrinsic desirability
        for (int p = 1; p <= P; p++) {
            int ww = rnd.next(2, 20);
            int vv = rnd.next(1, base);
            if (p == pref[j]) {
                // preferred pump: cheaper water, richer yield
                ww = max(1, ww - rnd.next(2, 8));
                vv = min(300, vv + rnd.next(60, 160));
            }
            w[p][j] = ww;
            v[p][j] = vv;
        }
    }

    // capacities: tight enough that not everything fits -> real packing problem.
    // average weight ~11; a full assignment would need ~ (B/P)*11 per pump.
    // load factor < 1 forces choices.
    double loadFactor = 0.45 + 0.03 * (testId % 3); // 0.45..0.51
    vector<int> cap(P + 1);
    for (int p = 1; p <= P; p++) {
        double avgLoad = (double)B / (double)P * 11.0 * loadFactor;
        int c = (int)llround(avgLoad) + rnd.next(-3, 6);
        if (c < 25) c = 25;              // ensure single blocks always fit
        cap[p] = c;
    }

    // ---- emit ----
    printf("%d %d\n", P, B);
    for (int p = 1; p <= P; p++) printf("%d%c", cap[p], p == P ? '\n' : ' ');
    for (int j = 1; j <= B; j++) {
        for (int p = 1; p <= P; p++) {
            printf("%d %d", w[p][j], v[p][j]);
            printf("%c", p == P ? '\n' : ' ');
        }
    }
    return 0;
}
