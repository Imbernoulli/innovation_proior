#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Salmon Migration Ladder: synthetic weighted Max-SAT generator.
// testId is a difficulty/structure ladder:
//   testId 1  -> tiny (example scale), testId 10 -> large & dense.
// We deliberately bias clause weight toward POSITIVE (high-flow) runs so the
// all-low-flow baseline is genuinely weak, while keeping a nontrivial mix of
// negative and mixed runs so no single extreme (all-low / all-high) is optimal.

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int n = 8 + 20 * (testId - 1);          // 8, 28, ..., 188  (gates / variables)
    double dens = 4.0 + 0.4 * testId;       // clauses per gate: 4.4 .. 8.0
    int m = (int)llround(n * dens);
    m = max(m, 4);

    // per-test structural knobs
    double pPos = 0.50 + 0.02 * (testId % 3);   // fraction of positive-only runs
    double pNeg = 0.32;                          // fraction of negative-only runs
    // remainder are mixed-sign runs.
    int wPosSmallHi = 15 + 2 * (testId % 4);    // small positive weight ceiling
    int wPosBigHi   = 120 + 10 * (testId % 3);  // heavy positive weight ceiling
    int wOtherHi    = 25 + 3 * (testId % 4);    // negative/mixed weight ceiling

    struct Clause { int w; vector<int> lits; };
    vector<Clause> cls;
    cls.reserve(m);

    auto pickVars = [&](int k) {
        // k distinct variables from 1..n
        k = min(k, n);
        vector<int> v;
        set<int> used;
        while ((int)v.size() < k) {
            int x = rnd.next(1, n);
            if (used.insert(x).second) v.push_back(x);
        }
        return v;
    };

    auto clauseSize = [&]() {
        int r = rnd.next(0, 99);
        if (r < 15) return 2;
        if (r < 80) return 3;
        if (r < 95) return 4;
        return 5;
    };

    // Force the FIRST run to be a negative-only run so the all-low-flow baseline
    // B is strictly positive regardless of the random draw.
    {
        Clause c;
        c.w = rnd.next(1, wOtherHi);
        int k = min(2, n);
        for (int x : pickVars(k)) c.lits.push_back(-x);
        cls.push_back(c);
    }

    for (int j = 1; j < m; j++) {
        Clause c;
        int k = clauseSize();
        vector<int> vs = pickVars(k);
        double roll = rnd.next(0.0, 1.0);
        if (roll < pPos) {
            // positive-only run, heavy-tailed weight (dominant weight mass)
            for (int x : vs) c.lits.push_back(+x);
            if (rnd.next(0, 99) < 78) c.w = rnd.next(1, wPosSmallHi);
            else                      c.w = rnd.next(wPosSmallHi, wPosBigHi);
        } else if (roll < pPos + pNeg) {
            // negative-only run (helps the low-flow baseline), light weight
            for (int x : vs) c.lits.push_back(-x);
            c.w = rnd.next(1, wOtherHi);
        } else {
            // mixed-sign run, light weight; guarantee >=1 of each sign
            int kk = (int)vs.size();
            int flipUpTo = rnd.next(1, kk - 1 >= 1 ? kk - 1 : 1);
            for (int i = 0; i < kk; i++) {
                int sgn = (i < flipUpTo) ? -1 : +1;
                c.lits.push_back(sgn * vs[i]);
            }
            shuffle(c.lits.begin(), c.lits.end());
            c.w = rnd.next(1, wOtherHi);
        }
        cls.push_back(c);
    }

    // shuffle run order so structure/index is not informative
    shuffle(cls.begin(), cls.end());

    printf("%d %d\n", n, (int)cls.size());
    for (auto& c : cls) {
        printf("%d %d", c.w, (int)c.lits.size());
        for (int l : c.lits) printf(" %d", l);
        printf("\n");
    }
    return 0;
}
