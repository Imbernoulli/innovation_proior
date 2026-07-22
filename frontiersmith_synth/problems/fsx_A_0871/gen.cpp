#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Generator for "Journeymen, Masters, and Sweeteners" (near-stable-welfare-forge).
//
// Emits: n m e lambda
//        then e lines: i j a b s     (1<=i<=n journeyman, 1<=j<=m master,
//                                      a = journeyman i's utility for master j,
//                                      b = master j's utility for journeyman i,
//                                      s = sweetener cap for pair (i,j))
// No duplicate (i,j) edges. testId is a difficulty/structure ladder:
//   1-2  tiny/small uniform random          (sanity)
//   3-5  medium uniform / mildly clustered  (baseline scaling)
//   6    PLANTED+TRAP: a few "contested star" masters, low caps on star edges
//   7    PLANTED good sub-assignment + stars (structure to be discovered)
//   8    NEEDLE: one very high value pair hidden in a sea of low-value noise
//   9    large multi-star adversarial, skewed caps
//   10   largest, fills the constraint envelope, heaviest multi-star + skew

struct Edge { int i, j, a, b, s; };

static void addEdge(vector<Edge>& edges, set<long long>& seen, int m, int i, int j, int a, int b, int s) {
    long long key = (long long)i * 1000000007LL + j;
    if (seen.count(key)) return;
    seen.insert(key);
    edges.push_back({i, j, a, b, s});
}

typedef long long ll;

// Simulates the checker's own baseline: greedy-in-input-order matching, zero
// payments. Returns (welfare, blocking-count).
static pair<ll, ll> simulateBaseline(int n, int m, const vector<Edge>& edges) {
    int e = (int)edges.size();
    vector<char> usedI(n + 1, 0), usedJ(m + 1, 0), isM(e, 0);
    vector<ll> u(n + 1, 0), v(m + 1, 0);
    for (int k = 0; k < e; k++) {
        if (!usedI[edges[k].i] && !usedJ[edges[k].j]) {
            usedI[edges[k].i] = usedJ[edges[k].j] = 1;
            isM[k] = 1;
            u[edges[k].i] = edges[k].a;
            v[edges[k].j] = edges[k].b;
        }
    }
    ll W = 0, K = 0;
    for (int k = 0; k < e; k++) if (isM[k]) W += edges[k].a + edges[k].b;
    for (int k = 0; k < e; k++) {
        if (isM[k]) continue;
        if (edges[k].a > u[edges[k].i] && edges[k].b > v[edges[k].j]) K++;
    }
    return {W, K};
}

// Simulates the "greedy" tier: weight-sorted (a+b desc) matching, zero payments.
static pair<ll, ll> simulateWeightGreedy(int n, int m, const vector<Edge>& edges) {
    int e = (int)edges.size();
    vector<int> order(e);
    for (int k = 0; k < e; k++) order[k] = k;
    sort(order.begin(), order.end(), [&](int x, int y) {
        int wx = edges[x].a + edges[x].b, wy = edges[y].a + edges[y].b;
        if (wx != wy) return wx > wy;
        return x < y;
    });
    vector<char> usedI(n + 1, 0), usedJ(m + 1, 0), isM(e, 0);
    vector<ll> u(n + 1, 0), v(m + 1, 0);
    for (int k : order) {
        if (!usedI[edges[k].i] && !usedJ[edges[k].j]) {
            usedI[edges[k].i] = usedJ[edges[k].j] = 1;
            isM[k] = 1;
            u[edges[k].i] = edges[k].a;
            v[edges[k].j] = edges[k].b;
        }
    }
    ll W = 0, K = 0;
    for (int k = 0; k < e; k++) if (isM[k]) W += edges[k].a + edges[k].b;
    for (int k = 0; k < e; k++) {
        if (isM[k]) continue;
        if (edges[k].a > u[edges[k].i] && edges[k].b > v[edges[k].j]) K++;
    }
    return {W, K};
}

// Self-calibrates lambda from the instance just generated. The PRIMARY driver is
// the baseline construction alone (retain ~40% of its raw welfare after the
// blocking penalty) -- this depends only on the market's own structure, not on
// what any particular solution strategy achieves, so the resulting difficulty
// varies organically from test to test instead of being reverse-engineered to a
// fixed target score. A safety ceiling (using the weight-sorted no-payment
// construction) only ever pushes lambda UP, and only when needed, to stop a
// dense/low-value instance from letting that construction blow past the 1.0
// score cap and erase differentiation from a payment-aware strategy.
static int calibrateLambda(int n, int m, const vector<Edge>& edges) {
    auto base = simulateBaseline(n, m, edges);
    auto grd = simulateWeightGreedy(n, m, edges);
    ll Wbase = base.first, Kbase = base.second, Wg = grd.first, Kg = grd.second;

    double retainFrac = 0.22; // baseline keeps ~22% of its raw welfare
    double lamF = (Kbase > 0) ? (1.0 - retainFrac) * (double)Wbase / (double)Kbase : 50.0;

    double ceilG = 0.85; // never let ANY zero-payment construction saturate near the cap
    if (Kg > 0) {
        double Braw = (double)Wbase - lamF * (double)Kbase; if (Braw < 1.0) Braw = 1.0;
        double Fg = (double)Wg - lamF * (double)Kg; if (Fg < 1.0) Fg = 1.0;
        double ratioG = Fg / (10.0 * Braw);
        if (ratioG > ceilG) {
            double denom = (double)Kg - 10.0 * ceilG * (double)Kbase;
            if (fabs(denom) > 1e-9) {
                double cand = ((double)Wg - 10.0 * ceilG * (double)Wbase) / denom;
                if (cand > lamF && isfinite(cand)) lamF = cand;
            }
        }
    }

    // hard safety nets: never crush the baseline below ~10% retention, and never
    // crush the weight-sorted construction's own welfare below ~15% of its raw value.
    double lamHardCapBase = (Kbase > 0) ? 0.90 * (double)Wbase / (double)Kbase : 2000.0;
    double lamHardCapG = (Kg > 0) ? 0.85 * (double)Wg / (double)Kg : 2000.0;
    lamF = min(lamF, min(lamHardCapBase, lamHardCapG));
    if (!(lamF > 0.0) || !isfinite(lamF)) lamF = 50.0;
    lamF = max(lamF, 1.0);
    int lambda = (int)llround(lamF);
    if (lambda < 1) lambda = 1;
    if (lambda > 2000) lambda = 2000;
    return lambda;
}

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int n, m;
    vector<Edge> edges;
    set<long long> seen;

    if (testId == 1) {
        n = 5; m = 5;
        int targetE = 14;
        while ((int)edges.size() < targetE) {
            int i = rnd.next(1, n), j = rnd.next(1, m);
            addEdge(edges, seen, m, i, j, rnd.next(50, 300), rnd.next(50, 300), rnd.next(10, 100));
        }
    } else if (testId == 2) {
        n = 8; m = 8;
        int targetE = 20;
        while ((int)edges.size() < targetE) {
            int i = rnd.next(1, n), j = rnd.next(1, m);
            addEdge(edges, seen, m, i, j, rnd.next(50, 500), rnd.next(50, 500), rnd.next(10, 150));
        }
    } else if (testId == 3) {
        n = 20; m = 18;
        int targetE = 65;
        while ((int)edges.size() < targetE) {
            int i = rnd.next(1, n), j = rnd.next(1, m);
            addEdge(edges, seen, m, i, j, rnd.next(1, 1000), rnd.next(1, 1000), rnd.next(0, 300));
        }
    } else if (testId == 4) {
        n = 60; m = 55;
        int targetE = 230;
        while ((int)edges.size() < targetE) {
            int i = rnd.next(1, n), j = rnd.next(1, m);
            addEdge(edges, seen, m, i, j, rnd.next(1, 1000), rnd.next(1, 1000), rnd.next(0, 350));
        }
    } else if (testId == 5) {
        // mildly clustered: each journeyman prefers a "home cluster" of masters
        n = 120; m = 110;
        int clusters = 10, csize = (m + clusters - 1) / clusters;
        for (int i = 1; i <= n; i++) {
            int home = rnd.next(0, clusters - 1);
            int deg = rnd.next(3, 6);
            for (int t = 0; t < deg; t++) {
                int j = home * csize + rnd.next(1, csize);
                if (j > m) j = rnd.next(1, m);
                addEdge(edges, seen, m, i, j, rnd.next(200, 900), rnd.next(200, 900), rnd.next(30, 400));
            }
        }
    } else if (testId == 6) {
        // TRAP: a few contested star masters everyone loves, low sweetener cap on
        // star edges; regular edges elsewhere have moderate value and roomy caps.
        n = 150; m = 140;
        int stars = 4;
        vector<int> starM;
        for (int s = 0; s < stars; s++) starM.push_back(rnd.next(1, m));
        for (int i = 1; i <= n; i++) {
            // most journeymen covet at least one star
            if (rnd.next(0, 99) < 65) {
                int j = starM[rnd.next(0, stars - 1)];
                addEdge(edges, seen, m, i, j, rnd.next(880, 1000), rnd.next(250, 980), rnd.next(0, 15));
            }
            int deg = rnd.next(2, 5);
            for (int t = 0; t < deg; t++) {
                int j = rnd.next(1, m);
                addEdge(edges, seen, m, i, j, rnd.next(200, 700), rnd.next(200, 700), rnd.next(80, 350));
            }
        }
    } else if (testId == 7) {
        // PLANTED good sub-assignment (high combined value, generous caps) buried
        // among stars + noise; the planted block should be prioritized by an
        // insightful matching, not merely stumbled on by weight-sort ties.
        n = 300; m = 280;
        int stars = 5;
        vector<int> starM;
        for (int s = 0; s < stars; s++) starM.push_back(rnd.next(1, m));
        int plantSz = 20;
        for (int k = 0; k < plantSz; k++) {
            int i = k + 1, j = k + 1;
            addEdge(edges, seen, m, i, j, rnd.next(700, 780), rnd.next(700, 780), rnd.next(250, 400));
        }
        for (int i = 1; i <= n; i++) {
            if (rnd.next(0, 99) < 55) {
                int j = starM[rnd.next(0, stars - 1)];
                addEdge(edges, seen, m, i, j, rnd.next(880, 1000), rnd.next(250, 980), rnd.next(0, 15));
            }
            int deg = rnd.next(2, 5);
            for (int t = 0; t < deg; t++) {
                int j = rnd.next(1, m);
                addEdge(edges, seen, m, i, j, rnd.next(150, 600), rnd.next(150, 600), rnd.next(60, 300));
            }
        }
    } else if (testId == 8) {
        // NEEDLE: one very high value, well-capped pair hidden inside a dense
        // field of low-value, tightly-capped noise edges.
        n = 260; m = 240;
        for (int i = 1; i <= n; i++) {
            int deg = rnd.next(6, 10);
            for (int t = 0; t < deg; t++) {
                int j = rnd.next(1, m);
                addEdge(edges, seen, m, i, j, rnd.next(1, 60), rnd.next(1, 60), rnd.next(0, 20));
            }
        }
        int ni = rnd.next(1, n), nj = rnd.next(1, m);
        seen.erase((long long)ni * 1000000007LL + nj);
        edges.erase(remove_if(edges.begin(), edges.end(), [&](const Edge& e){ return e.i == ni && e.j == nj; }), edges.end());
        addEdge(edges, seen, m, ni, nj, 1000, 1000, 500);
        // a handful of near-needle decoys (moderately high, still much weaker)
        for (int k = 0; k < 5; k++) {
            int i = rnd.next(1, n), j = rnd.next(1, m);
            addEdge(edges, seen, m, i, j, rnd.next(600, 820), rnd.next(600, 820), rnd.next(0, 30));
        }
    } else if (testId == 9) {
        n = 900; m = 850;
        int stars = 8;
        vector<int> starM;
        for (int s = 0; s < stars; s++) starM.push_back(rnd.next(1, m));
        for (int i = 1; i <= n; i++) {
            if (rnd.next(0, 99) < 50) {
                int j = starM[rnd.next(0, stars - 1)];
                addEdge(edges, seen, m, i, j, rnd.next(880, 1000), rnd.next(250, 980), rnd.next(0, 20));
            }
            int deg = rnd.next(6, 10);
            for (int t = 0; t < deg; t++) {
                int j = rnd.next(1, m);
                addEdge(edges, seen, m, i, j, rnd.next(150, 700), rnd.next(150, 700), rnd.next(50, 350));
            }
        }
    } else {
        // testId 10: largest, fills the envelope, heaviest multi-star + skew.
        n = 2000; m = 1900;
        int stars = 12;
        vector<int> starM;
        for (int s = 0; s < stars; s++) starM.push_back(rnd.next(1, m));
        for (int i = 1; i <= n; i++) {
            if (rnd.next(0, 99) < 45) {
                int j = starM[rnd.next(0, stars - 1)];
                addEdge(edges, seen, m, i, j, rnd.next(880, 1000), rnd.next(250, 980), rnd.next(0, 20));
            }
            int deg = rnd.next(7, 9);
            for (int t = 0; t < deg; t++) {
                int j = rnd.next(1, m);
                addEdge(edges, seen, m, i, j, rnd.next(150, 700), rnd.next(150, 700), rnd.next(50, 350));
            }
        }
    }

    int e = (int)edges.size();
    int lambda = calibrateLambda(n, m, edges);
    printf("%d %d %d %d\n", n, m, e, lambda);
    for (auto& ed : edges) printf("%d %d %d %d %d\n", ed.i, ed.j, ed.a, ed.b, ed.s);
    return 0;
}
