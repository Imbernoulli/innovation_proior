#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Skybridge Fulfillment: Drone Payload Dispatch (Generalized Assignment Problem).
// testId is a difficulty/structure ladder:
//   testId 1  -> tiny (example scale), grows to a larger, tighter, more skewed
//   instance by testId 10.  Each parcel fits alone on every drone (e_ij <= c_i),
//   so a feasible assignment always exists; total budget is deliberately tight so
//   only a fraction of parcels can be served -> real optimization / divergence.
int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int m = min(6, 2 + (testId - 1) / 2);          // 2,2,3,3,4,4,5,5,6,6
    int n = min(40, 4 + 4 * (testId - 1));          // 4,8,12,...,40

    int emax = 15;                                  // max per-drone energy cost

    // per-drone efficiency: some drones are systematically cheaper for a parcel.
    // cost e_ij = clamp( round(base_j * eff_ij), 1, emax )
    vector<double> eff(m);
    for (int i = 0; i < m; i++) eff[i] = rnd.next(0.55, 1.45);

    // value skew grows with testId: a few heavy parcels among many light ones.
    double heavyProb = 0.10 + 0.02 * (testId % 4);

    vector<int> w(n);
    vector<vector<int>> e(m, vector<int>(n));
    for (int j = 0; j < n; j++) {
        if (rnd.next(0.0, 1.0) < heavyProb)
            w[j] = rnd.next(60, 100);
        else
            w[j] = rnd.next(1, 40);
        double base = rnd.next(3.0, (double)emax);  // intrinsic difficulty of parcel j
        for (int i = 0; i < m; i++) {
            int v = (int)llround(base * eff[i]);
            if (v < 1) v = 1;
            if (v > emax) v = emax;
            e[i][j] = v;
        }
    }

    // Budgets: total budget targets ~ 3n..5n energy units while a naive serve-all
    // would need ~ sum of cheapest costs.  Tight -> contention.  Also ensure each
    // budget >= emax so every parcel fits alone on every drone.
    int per = max(emax, (int)llround(4.0 * (double)n / (double)m));
    vector<int> c(m);
    for (int i = 0; i < m; i++) {
        int lo = max(emax, (int)llround(per * 0.75));
        int hi = (int)llround(per * 1.25);
        if (hi < lo) hi = lo;
        c[i] = rnd.next(lo, hi);
    }

    printf("%d %d\n", m, n);
    for (int i = 0; i < m; i++) printf("%d%c", c[i], i + 1 == m ? '\n' : ' ');
    for (int j = 0; j < n; j++) {
        printf("%d", w[j]);
        for (int i = 0; i < m; i++) printf(" %d", e[i][j]);
        printf("\n");
    }
    return 0;
}
