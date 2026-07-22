#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Desert Temple of a Thousand Mirrors"  (generator)
// family: mirror-field-shading-ballet
//
// A temple tower stands at the origin. Candidate heliostat sites are a
// random subset of the nonzero integer lattice points (x,y) inside a disk
// of radius Rout. Sites inside the inner radius Rin survive with a HIGH,
// fixed probability (a dense "hot zone" around the tower); sites beyond Rin
// survive with a probability that decays further with radius -- a DENSITY
// GRADIENT baked into the input the solver must read (mechanism:
// density-gradient-layout). Each site's raw quality q also decays with
// radius (plus small noise), so the unique-per-mirror "sweet spot" is
// always near the tower.
//
// The day is swept by K discrete sun steps (mechanism: angular-sweep-coverage):
// each step k carries an integer ground-shadow offset (dx_k,dy_k) -- the
// direction/length (in lattice units, since the panel height cancels the sun's
// vertical component by construction) a mirror casts its shadow toward a
// neighbour "downsun" of it -- and an energy multiplier e_k. The offset is
// (0,0) at exact local noon (sun overhead -> no shadow, ever) and grows in a
// rotating compass direction toward both ends of the day (long low-angle
// shadows), while e_k is largest at noon and smallest at the extremes.
//
// TRAP: because quality strictly decays with radius, "pick the M
// highest-quality candidates" collapses onto the dense hot zone. The hot
// zone's lattice is dense enough that, at low-sun steps, many pairs of
// chosen sites sit EXACTLY (dx_k,dy_k) apart (mechanism: mutual-shading via
// exact ray incidence), and many chosen sites lie EXACTLY on the same ray
// from the tower as a closer chosen site (mutual blocking of the reflected
// beam toward the tower, permanently zeroing that mirror). Both effects are
// far more common in the crowded core -- exactly where the naive
// "sweet spot" greedy piles everything up -- than in the sparse periphery.
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;

    int Rin  = 4 + (int)llround(f * 10.0);              // 4 .. 14
    int Rout = Rin + 8 + (int)llround(f * 26.0);         // Rin+8 .. Rin+34
    int K = 7 + 2 * (testId % 3);                        // 7, 9, 11 (cycling, always odd)

    ll Rin2  = (ll)Rin * Rin;
    ll Rout2 = (ll)Rout * Rout;
    double QBASE = 100.0;
    double slope = (Rout > 0) ? (82.0 / Rout) : 0.0;

    struct Site { int x, y, q; };
    // Emit candidates in natural raster order across the whole disk. The
    // "first min(M,P) ids" baseline this induces therefore draws broadly
    // (mostly low-quality outer sites, since they vastly outnumber the
    // dense hot zone in raster order) while the quality-sorted greedy pulls
    // exclusively from the high-quality core -- a genuine quality gap that
    // the hot zone's mutual interference then eats back into.
    vector<Site> sites;
    sites.reserve((size_t)(3.2 * Rout * Rout) + 16);

    for (int y = -Rout; y <= Rout; y++) {
        for (int x = -Rout; x <= Rout; x++) {
            if (x == 0 && y == 0) continue;
            ll r2 = (ll)x * x + (ll)y * y;
            if (r2 > Rout2) continue;
            bool inHot = (r2 <= Rin2);
            bool keep;
            const double HOT_DENSITY = 0.50;
            if (inHot) {
                keep = (rnd.next(0.0, 1.0) < HOT_DENSITY);
            } else {
                double r = sqrt((double)r2);
                double band = (double)(Rout - Rin);
                double p = HOT_DENSITY * max(0.08, 1.0 - (r - Rin) / max(1.0, band));
                keep = (rnd.next(0.0, 1.0) < p);
            }
            if (!keep) continue;
            double r = sqrt((double)r2);
            int q = (int)llround(QBASE - slope * r) + rnd.next(-3, 3);
            if (q < 1) q = 1;
            sites.push_back({x, y, q});
        }
    }

    int P = (int)sites.size();
    // Should never happen given Rin>=4, but guard anyway.
    if (P < 4) { sites.push_back({1, 0, 90}); sites.push_back({0, 1, 88}); P = (int)sites.size(); }

    int Pin = 0;
    for (auto& s : sites) if ((ll)s.x * s.x + (ll)s.y * s.y <= Rin2) Pin++;
    if (Pin < 1) Pin = 1;

    double mfactor = 0.55 + 1.55 * f;   // 0.55 .. 2.10
    int M = (int)llround(Pin * mfactor);
    if (M < 1) M = 1;
    if (M > P) M = P;

    int mid = (K - 1) / 2;
    static const int cdx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    static const int cdy[8] = {0, 1, 1, 1, 0, -1, -1, -1};
    const int EBASE = 5, ESTEP = 3;

    printf("%d %d %d\n", P, M, K);
    for (auto& s : sites) printf("%d %d %d\n", s.x, s.y, s.q);
    for (int j = 0; j < K; j++) {
        int mag = abs(j - mid);
        int cidx = j % 8;
        int dx = cdx[cidx] * mag;
        int dy = cdy[cidx] * mag;
        int e = EBASE + (mid - mag) * ESTEP;
        printf("%d %d %d\n", dx, dy, e);
    }
    return 0;
}
