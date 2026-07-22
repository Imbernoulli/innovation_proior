#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// gen.cpp -- "Courier depots that survive every season".
// Seasonal depot siting scored on the WORST season's weighted nearest-depot cost.
// testId is a difficulty/structure ladder (tiny at 1 -> N=2500 at 10).
//
// STRUCTURE (the trap): each season k owns a geographically DISJOINT hot zone (a blob of
// addresses that are hot in season k and near-silent in every other season). Zones live on
// a circle around the board centre, so serving season k well REQUIRES a depot near zone k.
// Crucially, zones differ in SPREAD, not in total mass: some zones are compact (tiny radius,
// high address density) and some are spread (large radius, low density) but every zone
// carries the same total demand. Pooled k-means chases DENSITY, so it packs depots onto the
// compact zones and starves the spread zones -- yet a starved spread season becomes the
// worst season and blows up F. The insightful layout hedges: it weights addresses by their
// WORST-CASE (max-over-seasons) demand and pours extra depots into the spread zones to hold
// down the max. On most tests P >= K so every zone CAN be covered; the fight is allocation.
//
// Deterministic: all randomness comes from testlib `rnd`, seeded by testId.
int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    int idx = min(max(testId, 1), 10);

    const int C = 100000;
    int Ns[10] = {8,   40,  120, 300, 600, 900, 1200, 1600, 2000, 2500};
    int Ks[10] = {2,   3,   3,   4,   4,   5,   5,    6,    6,    6};
    // Many depots (P ~ 2.5*K): the fight is the ALLOCATION of depots across zones. A demand-
    // proportional recipe hands ~P/K depots to every zone regardless of geometry; but a SPREAD
    // zone needs several depots to be served cheaply while a COMPACT zone needs only one. So
    // the naive recipe over-serves the compact zones and starves the spread ones, whose season
    // becomes the worst. The insight allocates depots to equalise the worst season.
    int Ps[10] = {3,   7,   8,   10,  12,  14,  16,   18,   20,   24};
    int N = Ns[idx - 1];
    int K = Ks[idx - 1];
    int P = Ps[idx - 1];

    long long cx = C / 2, cy = C / 2;
    double Rc = C * 0.36;                 // radius of the ring of zone centres
    double ang0 = rnd.next(0.0, 2 * acos(-1.0));

    // zone centres on a ring (well separated)
    vector<long long> zx(K), zy(K);
    for (int k = 0; k < K; k++) {
        double a = ang0 + 2 * acos(-1.0) * k / K;
        zx[k] = (long long)llround(cx + Rc * cos(a));
        zy[k] = (long long)llround(cy + Rc * sin(a));
        zx[k] = min((long long)C, max(0LL, zx[k]));
        zy[k] = min((long long)C, max(0LL, zy[k]));
    }

    // The average-vs-worst trap. Zones split into two kinds:
    //   HEAVY + COMPACT (even k): large seasonal mass, tiny radius -> a depot parked here
    //     serves the whole season almost for free, and the big mass dominates the POOLED
    //     demand, so weighted/pooled k-means is magnetically drawn to these.
    //   LIGHT + SPREAD  (odd k): small seasonal mass, large radius -> ignored by the pooled
    //     recipe (low weight), yet when starved its addresses sit far from any depot and its
    //     season's cost mass*distance becomes the WORST season. The insightful layout spends
    //     a scarce depot to hedge this light-but-remote zone down.
    // Masses are UNEQUAL on purpose (this is the trap); the ratio is modest so no reference
    // solution saturates the score.
    // EQUAL mass and EQUAL address count per zone (so a demand-proportional recipe seeds every
    // zone the same), but very UNEQUAL spread: even zones are broad, odd zones are tight.
    // Serving a broad zone cheaply needs several depots; a tight zone needs one. The naive
    // even split therefore starves the broad zones -> their season is the worst.
    vector<double> rad(K);
    vector<double> massMul(K);
    double rCompact = C * 0.012;
    double rSpread  = C * 0.13;
    for (int k = 0; k < K; k++) {
        massMul[k] = 1.0;                                // equal seasonal mass everywhere
        if (idx == 1) rad[k] = C * 0.02;                // symmetric tiny example
        else rad[k] = (k % 2 == 0) ? rSpread : rCompact; // zone 0 is broad
    }

    // addresses per zone: (almost) equal so each season carries equal address count
    vector<int> cnt(K, N / K);
    for (int k = 0; k < N % K; k++) cnt[k]++;

    long long baseBudget = 40000;         // scaled by massMul[k] -> unequal seasonal mass
    int background = 1;                   // silent-season background demand

    vector<long long> X, Y;
    vector<vector<long long>> W;          // W[i][k]
    X.reserve(N); Y.reserve(N); W.reserve(N);

    for (int k = 0; k < K; k++) {
        int m = cnt[k];
        if (m <= 0) continue;
        // draw m addresses around zone centre k within radius rad[k]
        vector<pair<long long,long long>> pts;
        for (int j = 0; j < m; j++) {
            // uniform-ish in a square patch of half-width rad[k]
            long long dx = (long long)llround(rnd.next(-rad[k], rad[k]));
            long long dy = (long long)llround(rnd.next(-rad[k], rad[k]));
            long long x = min((long long)C, max(0LL, zx[k] + dx));
            long long y = min((long long)C, max(0LL, zy[k] + dy));
            pts.push_back({x, y});
        }
        // distribute the zone budget across its m addresses as their season-k demand.
        // base share + small noise, then normalise so the zone's season-k mass == budget.
        vector<long long> share(m);
        long long raw = 0;
        for (int j = 0; j < m; j++) { share[j] = rnd.next(50, 150); raw += share[j]; }
        long long zoneBudget = (long long)llround(baseBudget * massMul[k]);
        for (int j = 0; j < m; j++) {
            long long hot = zoneBudget * share[j] / max(1LL, raw);
            if (hot < 1) hot = 1;
            vector<long long> wv(K, background);
            wv[k] = hot;
            X.push_back(pts[j].first);
            Y.push_back(pts[j].second);
            W.push_back(wv);
        }
    }

    // shuffle address order so input order carries no hint about zones/seasons
    int NN = (int)X.size();
    vector<int> perm(NN);
    for (int i = 0; i < NN; i++) perm[i] = i;
    shuffle(perm.begin(), perm.end());

    printf("%d %d %d\n", NN, P, K);
    for (int t = 0; t < NN; t++) {
        int i = perm[t];
        printf("%lld %lld", X[i], Y[i]);
        for (int k = 0; k < K; k++) printf(" %lld", W[i][k]);
        printf("\n");
    }
    return 0;
}
