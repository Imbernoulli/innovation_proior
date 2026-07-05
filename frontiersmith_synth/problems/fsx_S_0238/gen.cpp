#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    int n, m;
    if (t <= 1) { n = 5; m = 12; }
    else {
        n = min(500, t * 50);
        m = min(3000, n * 6);
    }

    // Each requirement: weight, k literals over distinct variables.
    // A requirement is either "positive-leaning" (literals tend ON) or
    // "negative-leaning" (literals tend OFF). The mixture makes neither
    // all-ON nor all-OFF close to optimal, so real optimization is needed.
    struct Clause { int w; vector<int> lit; };
    vector<Clause> cs;
    cs.reserve(m);

    for (int j = 0; j < m; j++) {
        Clause c;
        c.w = rnd.next(1, 100);
        int k = (n >= 3) ? rnd.next(2, 3) : 2;
        if (k > n) k = n;
        // pick k distinct variables
        vector<int> vars;
        {
            set<int> used;
            while ((int)vars.size() < k) {
                int v = rnd.next(1, n);
                if (used.insert(v).second) vars.push_back(v);
            }
        }
        bool positiveLeaning = (rnd.next(0, 99) < 62); // ~62% positive-leaning
        for (int v : vars) {
            bool pos;
            if (positiveLeaning) pos = (rnd.next(0, 9) < 9);   // ON w.p. 0.9
            else                 pos = (rnd.next(0, 9) < 1);   // ON w.p. 0.1
            c.lit.push_back(pos ? v : -v);
        }
        cs.push_back(c);
    }

    // Guarantee the all-OFF baseline satisfies positive weight (B > 0):
    // if no clause is satisfied by all-OFF, force clause 0 to have a negated literal.
    auto allOffSat = [&](const Clause& c) {
        for (int L : c.lit) if (L < 0) return true; // NOT x -> true under all-OFF
        return false;
    };
    long long B = 0;
    for (auto& c : cs) if (allOffSat(c)) B += c.w;
    if (B == 0 && !cs.empty()) {
        cs[0].lit[0] = -abs(cs[0].lit[0]);
    }

    printf("%d %d\n", n, m);
    for (auto& c : cs) {
        printf("%d %d", c.w, (int)c.lit.size());
        for (int L : c.lit) printf(" %d", L);
        printf("\n");
    }
    return 0;
}
