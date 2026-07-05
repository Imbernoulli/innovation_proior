#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Windward Array: weighted Max-SAT instance generator.
// Structure ladder (testId 1..10):
//   - testId 1 is tiny (example scale); n and m grow to the small-scale caps.
//   - clause density (m/n) hovers near the Max-3SAT hard region.
//   - polarity is positive-biased so the all-0 baseline B is weak (leaving
//     genuine optimization headroom), but a controlled block of negative-only
//     requirements pulls sensors the other way -> real conflicts, so neither
//     all-0 nor all-1 is optimal and greedy / local-search strategies diverge.
//   - weights are heavy-tailed on a per-test-varying fraction of requirements.

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- size ladder ----
    int n = 6 + (testId - 1) * 16;                 // 6, 22, 38, ..., 150
    if (n > 200) n = 200;

    double density = 4.0 + 0.12 * ((testId * 7) % 9); // ~4.0 .. ~5.0
    int m = (int)llround(density * n);
    if (m < 4) m = 4;
    if (m > 1000) m = 1000;

    // per-test structural knobs
    double heavyProb = 0.10 + 0.03 * (testId % 4);  // fraction of heavy channels
    // group probabilities: positive-only / negative-only / mixed
    double pPos = 0.52 + 0.02 * (testId % 3);       // ~0.52 .. 0.56
    double pNeg = 0.20 + 0.02 * (testId % 2);       // ~0.20 .. 0.22

    auto pickWeight = [&]() -> int {
        if (rnd.next(0.0, 1.0) < heavyProb) return rnd.next(120, 1000); // heavy channel
        return rnd.next(1, 25);                                          // routine channel
    };

    // pick L distinct sensor indices in 1..n
    auto pickVars = [&](int L) {
        vector<int> vs;
        while ((int)vs.size() < L) {
            int v = rnd.next(1, n);
            if (find(vs.begin(), vs.end(), v) == vs.end()) vs.push_back(v);
        }
        return vs;
    };

    struct Clause { int w; vector<int> lits; };
    vector<Clause> cls;
    cls.reserve(m);

    bool anyNeg = false;
    for (int c = 0; c < m; c++) {
        // clause length: mostly 3, some 2, occasional 1
        int L;
        double r = rnd.next(0.0, 1.0);
        if (n >= 3 && r < 0.68) L = 3;
        else if (n >= 2 && r < 0.90) L = 2;
        else L = 1;
        if (L > n) L = n;

        vector<int> vs = pickVars(L);

        double g = rnd.next(0.0, 1.0);
        vector<int> lits;
        if (g < pPos) {
            // positive-only requirement (wants mode 1)
            for (int v : vs) lits.push_back(+v);
        } else if (g < pPos + pNeg) {
            // negative-only requirement (wants mode 0) -> conflict + feeds B
            for (int v : vs) { lits.push_back(-v); anyNeg = true; }
        } else {
            // mixed: each literal random sign, but guarantee >=1 negative when L>=2
            for (int v : vs) {
                int sgn = (rnd.next(0, 1) == 0) ? -1 : +1;
                lits.push_back(sgn * v);
            }
            if (L >= 2) {
                bool hasNeg = false;
                for (int x : lits) if (x < 0) hasNeg = true;
                if (!hasNeg) { lits[rnd.next(0, L - 1)] *= -1; }
            }
            for (int x : lits) if (x < 0) anyNeg = true;
        }

        cls.push_back({pickWeight(), lits});
    }

    // guarantee B >= 1: ensure at least one negative literal exists somewhere
    if (!anyNeg && !cls.empty()) {
        cls[0].lits[0] = -abs(cls[0].lits[0]);
    }

    // shuffle requirement order so index != structural position
    shuffle(cls.begin(), cls.end());

    printf("%d %d\n", n, (int)cls.size());
    for (auto& c : cls) {
        printf("%d %d", c.w, (int)c.lits.size());
        for (int x : c.lits) printf(" %d", x);
        printf("\n");
    }
    return 0;
}
