#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Weighted (partial) Max-SAT generator, "Caldera Watch" volcano-monitoring skin.
//
// Structure that makes the instance open-ended AND the dormant (all-low-power)
// baseline genuinely weak:
//   * "coverage" clauses are mostly-positive 2/3-clauses (all-low-power satisfies
//     almost none of them), so a good plan must turn many stations high-power;
//   * a fraction of stations carry negative-unit "keep-dormant" clauses (battery /
//     bandwidth contention) that all-low-power DOES satisfy -> this is the baseline;
//   * a small fraction of coverage clauses carry one negative literal, so the
//     instance is a genuinely general CNF (not monotone) and greedy is not optimal.
//
// testId is a difficulty ladder: testId 1 is tiny (example scale); size + density
// grow to a large caldera by testId 10.
int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int n;
    if (testId == 1) n = 6;
    else n = 200 * testId;                 // 400, 600, ..., 2000

    // clause budget: coverage clauses ~ density * n
    double density = 5.0 + 0.4 * testId;   // ~5.4 .. ~9.0 coverage clauses per var
    int mCover = (int)(density * n);
    if (testId == 1) mCover = 6;

    // weight skew varies per test (uniform vs heavy-tailed importance)
    int wHi = 10 + 2 * (testId % 6);       // 10..20 importance ceiling
    // fraction of stations that get a "keep-dormant" negative-unit clause
    double negUnitProb = 0.35 + 0.03 * (testId % 5);   // 0.35 .. ~0.47
    int negUnitWHi = 5 + (testId % 4) * 2;             // 5..11
    // fraction of coverage clauses carrying exactly one negative literal
    double mixProb = 0.08 + 0.02 * (testId % 4);       // 0.08 .. 0.14

    struct Clause { int w; vector<int> lits; };
    vector<Clause> cls;

    auto randW = [&]() { return rnd.next(1, wHi); };

    // ---- negative-unit "keep dormant" clauses (define the baseline weight) ----
    for (int i = 1; i <= n; i++) {
        if (rnd.next(0.0, 1.0) < negUnitProb) {
            Clause c;
            c.w = rnd.next(1, negUnitWHi);
            c.lits = { -i };
            cls.push_back(c);
        }
    }

    // ---- coverage clauses (mostly positive, length 2 or 3) ----
    for (int e = 0; e < mCover; e++) {
        int L = (testId == 1) ? 2 : (rnd.next(0, 2) == 0 ? 2 : 3);
        if (L > n) L = n;
        // pick L distinct variables
        set<int> pick;
        while ((int)pick.size() < L) pick.insert(rnd.next(1, n));
        Clause c;
        c.w = randW();
        vector<int> vars(pick.begin(), pick.end());
        shuffle(vars.begin(), vars.end());
        // at most one negated literal per coverage clause
        int negPos = -1;
        if (rnd.next(0.0, 1.0) < mixProb) negPos = rnd.next(0, L - 1);
        for (int j = 0; j < L; j++) {
            int v = vars[j];
            c.lits.push_back(j == negPos ? -v : v);
        }
        cls.push_back(c);
    }

    // guarantee at least one negative literal exists (baseline >= 1)
    bool hasNeg = false;
    for (auto& c : cls) for (int l : c.lits) if (l < 0) hasNeg = true;
    if (!hasNeg) {
        Clause c; c.w = rnd.next(1, negUnitWHi); c.lits = { -rnd.next(1, n) };
        cls.push_back(c);
    }

    // shuffle clause order so index != structural role
    shuffle(cls.begin(), cls.end());

    int m = (int)cls.size();
    printf("%d %d\n", n, m);
    for (auto& c : cls) {
        printf("%d %d", c.w, (int)c.lits.size());
        for (int l : c.lits) printf(" %d", l);
        printf("\n");
    }
    return 0;
}
