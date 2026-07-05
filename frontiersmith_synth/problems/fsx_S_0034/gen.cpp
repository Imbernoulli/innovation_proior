#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Weighted Max-SAT skinned as a carnival ride-circuit tuning instance.
// Structure ladder: testId 1 tiny (example scale) -> large/adversarial by testId 10.
//
// Clause polarity mix is chosen so the all-reverse (all-zero) baseline is genuinely
// weak (leaving headroom to improve) while neither all-forward nor all-reverse is
// good, so the problem stays a real Max-SAT search:
//   - "preference" clauses (positive-only literals): all-reverse never satisfies them.
//   - "safety" clauses     (negative-only literals): all-reverse satisfies them.
//   - "mixed" clauses      (random signs): create genuine conflict.

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    // size ramp: n small at t=1, large by t=10
    int n = 8 * t * t + 6;              // 14, 38, 78, ..., ~806
    int m = 14 * n;                     // clauses

    // clause size varies per test for structural variety (2..4)
    int kbase = 2 + (t % 3);           // 2,3,4 cycling
    if (kbase < 2) kbase = 2;

    // polarity fractions (keep the all-reverse baseline weak but non-trivial)
    double fa = 0.58;                   // positive-only ("preference")
    double fb = 0.20;                   // negative-only ("safety")
    // remainder -> mixed

    // weight model: some tests uniform, some heavy-tailed (skewed guest groups)
    bool skewed = (t % 3 == 0);
    int wlo = 1, whi = 5 + (t % 4) * 3; // 5..14 uniform ceiling

    vector<array<int,2>> lits;          // reused buffer (var, sign)
    // pick k distinct variables
    auto pickVars = [&](int k, vector<int>& out) {
        out.clear();
        if (k >= n) { for (int v = 1; v <= n; v++) out.push_back(v); return; }
        // reservoir-free: sample distinct via set for small k
        set<int> s;
        while ((int)s.size() < k) s.insert(rnd.next(1, n));
        for (int v : s) out.push_back(v);
    };

    // build clauses into a buffer first so we can print efficiently
    string buf;
    buf.reserve((size_t)m * 8);
    char line[256];

    vector<int> vars;
    for (int c = 0; c < m; c++) {
        int k = kbase;
        if (k > n) k = n;
        pickVars(k, vars);

        int w;
        if (skewed) {
            if (rnd.next(0, 9) == 0) w = rnd.next(200, 1000); // rare whale group
            else w = 1;
        } else {
            w = rnd.next(wlo, whi);
        }

        double r = rnd.next(0.0, 1.0);
        int kind; // 0 positive-only, 1 negative-only, 2 mixed
        if (r < fa) kind = 0;
        else if (r < fa + fb) kind = 1;
        else kind = 2;

        int len = snprintf(line, sizeof(line), "%d %d", w, k);
        buf.append(line, len);
        for (int v : vars) {
            int sign;
            if (kind == 0) sign = +1;
            else if (kind == 1) sign = -1;
            else sign = (rnd.next(0, 1) ? +1 : -1);
            int lit = sign * v;
            len = snprintf(line, sizeof(line), " %d", lit);
            buf.append(line, len);
        }
        buf.push_back('\n');
    }

    printf("%d %d\n", n, m);
    fputs(buf.c_str(), stdout);
    return 0;
}
