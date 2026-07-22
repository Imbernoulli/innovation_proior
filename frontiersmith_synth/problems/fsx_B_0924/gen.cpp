#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// -----------------------------------------------------------------------------
// "Grow a Sharp Crystal, Not a Blob"  (generator)  family: anisotropic-eden-facet
//
// Emits:  SEED K M
//         r_0 r_1 ... r_{K-1}
// where the target region is the star-shaped polygon with vertex i at angle
// 2*pi*i/K, distance r_i from the origin, and M = number of lattice cells whose
// ray from the origin exits the polygon at or beyond the cell (computed here by
// direct bounding-box enumeration, using the SAME boundaryRadius() the checker
// and the strong reference use, so M is exactly consistent everywhere).
//
// Ladder across testId 1..10: tests 1-2 are near-regular (mild anisotropy,
// isotropic growth already covers them tolerably -- calibration). Tests 3..10
// plant a growing number of NEEDLE spikes (short "facet" base radius, a few very
// long "tip" vertices) with increasing spike/base ratio -- an isotropic blob of
// the same mass structurally cannot reach the tips (its radius is fixed by area
// alone), so these are the traps where anisotropic-attachment aimed at the true
// Wulff silhouette wins big, and a naive point-sample of the target radius still
// falls well short of a solver that properly integrates radius^2 over each bin.
// -----------------------------------------------------------------------------

static double boundaryRadius(int K, const vector<double>& r, double theta) {
    double sector = 2.0 * M_PI / K;
    int i = (int)floor(theta / sector);
    if (i < 0) i = 0;
    if (i >= K) i = K - 1;
    int j = (i + 1) % K;
    double th_i = i * sector, th_j = (i + 1) * sector;
    double ax = r[i] * cos(th_i), ay = r[i] * sin(th_i);
    double bx = r[j] * cos(th_j), by = r[j] * sin(th_j);
    double dx = cos(theta), dy = sin(theta);
    double abx = bx - ax, aby = by - ay;
    double denom = dx * aby - dy * abx;
    if (fabs(denom) < 1e-12) denom = (denom < 0 ? -1e-12 : 1e-12);
    double t = (ax * by - ay * bx) / denom;
    if (t < 1e-6) t = 1e-6;
    return t;
}

static long long countMass(int K, const vector<double>& r) {
    double rmax = 0;
    for (double v : r) rmax = max(rmax, v);
    int R = (int)ceil(rmax) + 2;
    long long cnt = 0;
    for (int dx = -R; dx <= R; dx++) {
        for (int dy = -R; dy <= R; dy++) {
            double rho = sqrt((double)dx * dx + (double)dy * dy);
            if (rho < 1e-9) { cnt++; continue; }
            double theta = atan2((double)dy, (double)dx);
            if (theta < 0) theta += 2 * M_PI;
            double bnd = boundaryRadius(K, r, theta);
            if (rho <= bnd + 1e-9) cnt++;
        }
    }
    return cnt;
}

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int K;
    vector<double> r;

    auto jitter = [&](int lo, int hi) { return lo + rnd.next(0, hi - lo); };

    auto plantSpikes = [&](int nSpikes, int spikeLo, int spikeHi) {
        vector<int> idx(K);
        for (int i = 0; i < K; i++) idx[i] = i;
        for (int i = (int)idx.size() - 1; i > 0; i--) swap(idx[i], idx[rnd.next(0, i)]);
        for (int t = 0; t < nSpikes; t++) r[idx[t]] = jitter(spikeLo, spikeHi);
    };

    switch (testId) {
        case 1: {  // tiny near-regular hexagon (calibration, no trap)
            K = 6;
            for (int i = 0; i < K; i++) r.push_back(jitter(5, 6));
            break;
        }
        case 2: {  // small near-regular hexagon (calibration, no trap)
            K = 6;
            for (int i = 0; i < K; i++) r.push_back(jitter(8, 10));
            break;
        }
        case 3: {  // 10-gon, thin base + 2 tall needle spikes (trap)
            K = 10;
            for (int i = 0; i < K; i++) r.push_back(jitter(6, 9));
            plantSpikes(2, 55, 70);
            break;
        }
        case 4: {  // 12-gon, thin base + 2 tall needle spikes (trap)
            K = 12;
            for (int i = 0; i < K; i++) r.push_back(jitter(6, 9));
            plantSpikes(2, 65, 80);
            break;
        }
        case 5: {  // 12-gon, thin base + 2 taller needle spikes (trap)
            K = 12;
            for (int i = 0; i < K; i++) r.push_back(jitter(6, 9));
            plantSpikes(2, 75, 95);
            break;
        }
        case 6: {  // 14-gon, thin base + 2 tall needle spikes (trap)
            K = 14;
            for (int i = 0; i < K; i++) r.push_back(jitter(6, 9));
            plantSpikes(2, 85, 105);
            break;
        }
        case 7: {  // 14-gon, thin base + 3 tall needle spikes, larger scale (trap)
            K = 14;
            for (int i = 0; i < K; i++) r.push_back(jitter(6, 9));
            plantSpikes(3, 95, 115);
            break;
        }
        case 8: {  // 16-gon, thin base + 3 tall needle spikes (extreme trap)
            K = 16;
            for (int i = 0; i < K; i++) r.push_back(jitter(5, 8));
            plantSpikes(3, 100, 120);
            break;
        }
        case 9: {  // 16-gon, 4 needle spikes, largest scale (extreme trap)
            K = 16;
            for (int i = 0; i < K; i++) r.push_back(jitter(5, 8));
            plantSpikes(4, 110, 130);
            break;
        }
        case 10: {  // 16-gon, thinnest base + 4 tallest spikes (hardest, fills envelope)
            K = 16;
            for (int i = 0; i < K; i++) r.push_back(jitter(5, 8));
            plantSpikes(4, 120, 150);
            break;
        }
        default: {
            K = 6;
            for (int i = 0; i < K; i++) r.push_back(jitter(8, 10));
        }
    }

    long long M = countMass(K, r);
    long long seed = rnd.next(1, 2000000000);

    printf("%lld %d %lld\n", seed, K, M);
    for (int i = 0; i < K; i++) printf("%lld%c", (long long)llround(r[i]), i + 1 == K ? '\n' : ' ');
    return 0;
}
