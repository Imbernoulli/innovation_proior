#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Riverbank Sound Festival: weighted Max-SAT (two-stage billing).
//
// Structure baked in so that the all-Side-Stage baseline is WEAK but the instance
// is a genuinely hard weighted Max-SAT:
//   * "billing demands": positive unit clauses (+i, weight W_UNIT) -- every act wants
//     the Main Stage. The all-Side (all-zero) baseline satisfies NONE of these, so it
//     is a poor reference and there is lots of head-room.
//   * "rivalry conflicts": binary clauses (-i OR -j, weight W_CONF) -- two rival acts
//     should not both headline the Main Stage. These create frustration (an
//     independent-set / max-cut style tension) so no constant placement is optimal.
//   * a sprinkle of general 3..5-literal mixed-sign clauses to keep it real Max-SAT
//     rather than pure MWIS, so per-act greedy and local search genuinely diverge.
//
// testId is a difficulty ladder: 1 tiny, 10 large / dense.

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ----- size ladder -----
    double frac = (testId - 1) / 9.0;              // 0 .. 1
    int n = (int)round(5.0 * pow(4000.0 / 5.0, frac));   // 5 .. 4000
    if (n < 3) n = 3;
    if (n > 4000) n = 4000;

    // conflict density varies with testId parity for structural diversity
    // (sparser on odd tests, denser on even tests -> different hardness / head-room)
    double confPerNode = (testId % 2 == 0) ? 3.0 : 1.6;
    // number of extra general mixed clauses
    double mixPerNode = 0.4 + 0.3 * frac;

    long long unitCnt = n;                          // one positive-unit per act
    long long confCnt = (long long)llround(confPerNode * n);
    long long mixCnt  = (long long)llround(mixPerNode  * n);

    // cap total clauses to stay within limits
    long long total = unitCnt + confCnt + mixCnt;
    long long CAP = 40000;
    if (total > CAP) {
        double s = (double)(CAP - unitCnt) / (double)(confCnt + mixCnt);
        if (s < 0) s = 0;
        confCnt = (long long)floor(confCnt * s);
        mixCnt  = (long long)floor(mixCnt  * s);
        total = unitCnt + confCnt + mixCnt;
    }
    if (total > CAP) total = CAP;  // safety

    // Build all clauses in a buffer first (so we can shuffle output order),
    // each clause = weight + list of literals.
    struct Clause { int w; vector<int> lits; };
    vector<Clause> cls;
    cls.reserve((size_t)total + 4);

    // 1) positive unit "billing" clauses (all-zero baseline satisfies NONE)
    for (long long c = 0; c < unitCnt; c++) {
        int i = (int)(c % n) + 1;
        int w = rnd.next(8, 40);
        cls.push_back({w, {+i}});
    }

    // 2) rivalry conflict clauses  (-i OR -j) : all-zero baseline satisfies ALL of them,
    //    which keeps B > 0; all-ones satisfies none of them (frustration).
    for (long long c = 0; c < confCnt; c++) {
        int i = rnd.next(1, n);
        int j = rnd.next(1, n);
        while (j == i) j = rnd.next(1, n);
        int w = rnd.next(2, 12);
        cls.push_back({w, {-i, -j}});
    }

    // 3) general mixed-sign clauses of length 3..5 (keep it real Max-SAT)
    for (long long c = 0; c < mixCnt; c++) {
        int k = rnd.next(3, 5);
        if (k > n) k = n;
        // pick k distinct variables
        set<int> chosen;
        while ((int)chosen.size() < k) chosen.insert(rnd.next(1, n));
        int w = rnd.next(2, 15);
        vector<int> lits;
        for (int v : chosen) {
            int sign = (rnd.next(0, 1) == 0) ? -1 : +1;  // balanced signs
            lits.push_back(sign * v);
        }
        cls.push_back({w, lits});
    }

    // Guarantee at least one clause satisfied by all-zero (safety for B >= 1):
    // add a single negative unit on act 1.
    cls.push_back({1, {-1}});

    // shuffle clause order deterministically
    shuffle(cls.begin(), cls.end());

    int m = (int)cls.size();
    printf("%d %d\n", n, m);
    for (auto& cl : cls) {
        printf("%d %d", (int)cl.lits.size(), cl.w);
        for (int L : cl.lits) printf(" %d", L);
        printf("\n");
    }
    return 0;
}
