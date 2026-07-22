#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Living-Hinge Curvature Fit"  (generator)  family: ligament-compliance-hinge
//
// A strip of m columns, each column starting with k parallel load-bearing
// ligaments.  Slitting (removing) ligaments in column x lowers its remaining
// net section r_x and raises its local bending compliance under a fixed applied
// moment M via the CUBIC law  curvature(r) = M / r^3.  A net-section strength
// floor S forbids r_x < S.  A single shared slit budget C bounds the total
// number of ligaments that may be removed across the whole strip.
//
// Because curvature(r) = M/r^3 is CONVEX and its slope is small near r=k (full
// material) and huge near r=S (near the floor), a solver that estimates "cuts
// needed" from the TANGENT line at r=k (the natural "more cuts = proportionally
// more bend" intuition) systematically OVER-estimates the cuts a column needs,
// burns the shared budget faster than necessary, and drives many columns all
// the way down to the strength floor -- "hits the strength wall first" -- while
// starving other columns of any budget at all.  The true minimum-cost column
// fit uses the exact inverse r = round((M/target)^(1/3)); over the SHARED
// budget the optimal allocation is a multiple-choice knapsack over all m
// columns, not a per-column greedy pass.
//
// TestId ladder (1..10): size grows with testId.  Three target-generation
// modes are planted across the ladder:
//   "rand"   - each column's target sits near an independently sampled
//              achievable curvature (with jitter), tests 1,2,4,9.
//   "trap"   - bimodal targets: ~half the columns need near-floor curvature,
//              ~half need almost none, under a TIGHT shared budget, tests
//              3,5,7,10.  The tangent-line greedy spends nearly the whole
//              budget flooring the first few "needy" columns it processes.
//   "needle" - a small minority of columns hide a large curvature need among
//              a majority of already-near-baseline columns, tests 6,8. The
//              tangent estimate makes the needle LOOK far more expensive than
//              its true (cubic-accelerated) cost, so it gets mis-prioritised.
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;

    int m = 6 + (int)llround(f * 154.0);      // 6 .. 160
    int k = 8 + (int)llround(f * 26.0);       // 8 .. 34

    double sfracTbl[3] = {0.15, 0.25, 0.35};
    double sfrac = sfracTbl[testId % 3];
    int S = max(2, (int)llround(k * sfrac));
    if (S > k - 4) S = max(2, k - 4);

    int factor = 1 + (testId % 3);            // 1,2,3
    ll M = (ll)factor * (ll)k * (ll)k * (ll)k;

    const char* modeTbl[10] = {"rand","rand","trap","rand","trap","needle","trap","needle","rand","trap"};
    string mode = modeTbl[testId - 1];
    double cfracTbl[10] = {0.70,0.65,0.28,0.55,0.25,0.30,0.30,0.32,0.50,0.22};
    double cfrac = cfracTbl[testId - 1];

    int maxU = k - S;
    ll costCap = (ll)m * (ll)maxU;
    ll C = max(1LL, (ll)llround(cfrac * (double)costCap));
    if (C > costCap) C = costCap;

    double curvK = (double)M / ((double)k * k * k);

    vector<ll> target(m);
    if (mode == "rand") {
        for (int i = 0; i < m; i++) {
            int rpref = rnd.next(S, k);
            double base = (double)M / ((double)rpref * rpref * rpref);
            double jitter = (rnd.next(-100, 100)) / 1000.0;   // +-10%
            double kappa = base * (1.0 + jitter);
            if (kappa < 0) kappa = 0;
            target[i] = (ll)llround(kappa * 1000.0);
        }
    } else if (mode == "trap") {
        // Bimodal targets under a TIGHT shared budget: about half the columns
        // sit at the near-floor extreme (large curvature need), half at the
        // near-full-material extreme (little to no need). A small offset band
        // keeps the reference point off the exact extreme sometimes, so the
        // true minimal-cost fit is occasionally interior -- one more place
        // the r=k tangent line mispredicts the true cubic curvature.
        int band = max(1, maxU / 6);
        for (int i = 0; i < m; i++) {
            int rpref = (rnd.next(0, 1) == 0) ? (S + rnd.next(0, band))
                                               : (k - rnd.next(0, band));
            double base = (double)M / ((double)rpref * rpref * rpref);
            double jitter = (rnd.next(-50, 50)) / 1000.0;     // +-5%
            double kappa = base * (1.0 + jitter);
            if (kappa < 0) kappa = 0;
            target[i] = (ll)llround(kappa * 1000.0);
        }
    } else { // needle
        int needles = max(1, m / 12);
        vector<int> idx(m);
        for (int i = 0; i < m; i++) idx[i] = i;
        for (int i = 0; i < m; i++) swap(idx[i], idx[rnd.next(i, m - 1)]);
        set<int> needleSet(idx.begin(), idx.begin() + needles);
        int band = max(1, maxU / 10);
        for (int i = 0; i < m; i++) {
            int rpref = needleSet.count(i) ? (S + rnd.next(0, band))
                                            : (k - rnd.next(0, band));
            double base = (double)M / ((double)rpref * rpref * rpref);
            double jitter = (rnd.next(-30, 30)) / 1000.0;     // +-3%
            double kappa = base * (1.0 + jitter);
            if (kappa < 0) kappa = 0;
            target[i] = (ll)llround(kappa * 1000.0);
        }
    }
    (void)curvK;

    printf("%d %d %d %lld %lld\n", m, k, S, M, C);
    for (int i = 0; i < m; i++) printf("%lld%c", target[i], (i + 1 == m) ? '\n' : ' ');
    return 0;
}
