#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Benchmark Stakes on the Mountain Road"   family: hotspot-rank-sampler
//
// Builds a weighted query trace (map<pos,count>) as a union of "hotspot"
// clusters (query-hotspot-mining) plus low-weight background noise, then
// emits  L K Q W  followed by Q ascending (pos,count) lines.
//
// Clusters are spaced across a bounded position band (so no single cluster's
// baseline contribution swamps the rest -- contribution ~ weight * position,
// and positions only vary by a small factor within a test) and given
// comparable total weight, so covering a genuinely useful SUBSET of them
// (K is deliberately well below the cluster count) is a real combinatorial
// choice, not "grab the one obviously dominant hotspot".
//
// The intended optimum places each of the K stakes at the LEFT edge of the
// group of trace positions it serves (asymmetric / one-sided distance: a
// stake only helps queries at or beyond it), grouped by an optimal weighted
// partition of the sorted trace (space-probe-duality: cost is a W-quantized
// probe count, not raw distance). The "textbook" trap is symmetric k-median:
// split into equal-weight buckets and plant each stake at the bucket's
// WEIGHTED MEDIAN (correct for symmetric L1 cost, wrong here). Several
// clusters are made WIDE and near-uniform so this median-vs-left-edge gap is
// large; needle spikes test that a solver doesn't dilute a high-value single
// position into a bucket average.
// -----------------------------------------------------------------------------

static void addClusterAt(map<ll,ll> &mp, ll L, ll center, ll halfwidth,
                          int npoints, ll totalWeight, int jitterPct) {
    ll lo = max(1LL, center - halfwidth), hi = min(L, center + halfwidth);
    if (hi <= lo) hi = min(L, lo + 1);
    set<ll> chosen;
    int tries = 0;
    while ((int)chosen.size() < npoints && tries < npoints * 40 + 200) {
        tries++;
        ll p = lo + rnd.next(0LL, max(0LL, hi - lo));
        chosen.insert(p);
    }
    if (chosen.empty()) return;
    ll base = max(1LL, totalWeight / (ll)chosen.size());
    ll jitter = max(0LL, base * jitterPct / 100);
    for (ll p : chosen) {
        ll wgt = base;
        if (jitter > 0) wgt += rnd.next(-jitter, jitter);
        if (wgt < 1) wgt = 1;
        mp[p] += wgt;
    }
}

static void addNeedle(map<ll,ll> &mp, ll L, ll pos, ll weight) {
    pos = max(1LL, min(L, pos));
    mp[pos] += weight;
}

static void addBackground(map<ll,ll> &mp, ll L, int n, ll wlo, ll whi) {
    for (int i = 0; i < n; i++) {
        ll p = 1 + rnd.next(0LL, max(0LL, L - 1));
        mp[p] += wlo + rnd.next(0LL, max(0LL, whi - wlo));
    }
}

// Places NCLUS clusters IRREGULARLY (randomized spacing, not an evenly
// spaced grid -- real drifting hotspots don't line up on a ruler, and a
// grid would let blind uniform spacing accidentally match the data) across
// the band [loFrac,hiFrac]*L, each carrying ~baseWeight total (so no
// cluster's baseline share is huge relative to the others), a subset
// (wideIdx, indexed by RANK along the road after sorting) made WIDE/
// near-uniform to stress the median-vs-left-edge placement gap.
static void buildClusters(map<ll,ll> &mp, ll L, int NCLUS, double loFrac, double hiFrac,
                           int npoints, ll baseWeight, const set<int> &wideIdx,
                           double narrowFracL, double wideFracL) {
    ll lo = (ll)(loFrac * L), hi = (ll)(hiFrac * L);
    if (hi <= lo) hi = lo + 1;

    // randomized fractions with a guaranteed minimum gap, so clusters keep
    // an irregular (not grid-like) spacing along the road.
    vector<double> raw(NCLUS);
    for (int i = 0; i < NCLUS; i++) raw[i] = rnd.next(0.0, 1.0);
    sort(raw.begin(), raw.end());
    double minGap = (NCLUS > 1) ? 0.55 / NCLUS : 0.0;
    double span = max(0.0, 1.0 - minGap * (NCLUS - 1));

    for (int i = 0; i < NCLUS; i++) {
        double frac = (NCLUS == 1) ? 0.5 : raw[i] * span + i * minGap;
        ll center = lo + (ll)((hi - lo) * frac);
        center = max(1LL, min(L, center));
        bool wide = wideIdx.count(i) > 0;
        ll halfwidth = max(1LL, (ll)((wide ? wideFracL : narrowFracL) * L));
        int npts = wide ? npoints * 2 : npoints;
        ll w = wide ? baseWeight : (ll)(baseWeight * (0.7 + 0.7 * rnd.next(0.0, 1.0)));
        addClusterAt(mp, L, center, halfwidth, npts, w, wide ? 10 : 25);
    }
}

int main(int argc, char *argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    ll L, W;
    int K, NCLUS;
    map<ll,ll> mp;

    switch (testId) {
        case 1: {
            L = 1500; W = 4; NCLUS = 5; K = 3;
            buildClusters(mp, L, NCLUS, 0.15, 0.90, 6, 900, {}, 0.01, 0.22);
            addBackground(mp, L, 8, 1, 3);
            break;
        }
        case 2: {
            L = 5000; W = 6; NCLUS = 6; K = 4;
            buildClusters(mp, L, NCLUS, 0.12, 0.92, 8, 1400, {}, 0.008, 0.20);
            addBackground(mp, L, 15, 1, 3);
            break;
        }
        case 3: {
            L = 12000; W = 8; NCLUS = 7; K = 6;
            buildClusters(mp, L, NCLUS, 0.10, 0.93, 10, 1700, {}, 0.006, 0.18);
            addBackground(mp, L, 25, 1, 4);
            break;
        }
        case 4: {   // TRAP: multiple WIDE uniformly-spread hotspots, fine W
            L = 30000; W = 5; NCLUS = 8; K = 6;
            buildClusters(mp, L, NCLUS, 0.10, 0.94, 12, 2000, {1, 4, 6}, 0.005, 0.15);
            addBackground(mp, L, 40, 1, 2);
            break;
        }
        case 5: {   // many drifting hotspots -> partition/selection matters
            L = 60000; W = 6; NCLUS = 9; K = 5;
            buildClusters(mp, L, NCLUS, 0.08, 0.95, 14, 2200, {}, 0.005, 0.13);
            addBackground(mp, L, 55, 1, 3);
            break;
        }
        case 6: {   // TRAP: needle spike + wide hotspot mix, scant budget
            L = 100000; W = 7; NCLUS = 8; K = 6;
            buildClusters(mp, L, NCLUS, 0.10, 0.92, 15, 2500, {2, 5}, 0.005, 0.14);
            addNeedle(mp, L, (ll)(0.63 * L), 9000);
            addBackground(mp, L, 60, 1, 3);
            break;
        }
        case 7: {
            L = 250000; W = 10; NCLUS = 10; K = 7;
            buildClusters(mp, L, NCLUS, 0.08, 0.95, 18, 2800, {}, 0.004, 0.12);
            addBackground(mp, L, 80, 1, 4);
            break;
        }
        case 8: {   // needle + moderate clusters at larger scale
            L = 500000; W = 15; NCLUS = 10; K = 8;
            buildClusters(mp, L, NCLUS, 0.08, 0.95, 20, 3200, {}, 0.004, 0.12);
            addNeedle(mp, L, (ll)(0.55 * L), 12000);
            addBackground(mp, L, 100, 1, 4);
            break;
        }
        case 9: {
            L = 1000000; W = 20; NCLUS = 12; K = 6;
            buildClusters(mp, L, NCLUS, 0.06, 0.96, 22, 3600, {}, 0.0035, 0.11);
            addNeedle(mp, L, (ll)(0.80 * L), 14000);
            addBackground(mp, L, 140, 1, 5);
            break;
        }
        default: {  // 10: largest, fills the envelope
            L = 2000000; W = 25; NCLUS = 14; K = 11;
            buildClusters(mp, L, NCLUS, 0.05, 0.97, 24, 4200, {0, 4, 8, 12}, 0.003, 0.10);
            addNeedle(mp, L, (ll)(0.70 * L), 16000);
            addBackground(mp, L, 180, 1, 5);
            break;
        }
    }

    int Q = (int)mp.size();
    if (K >= Q) K = max(1, Q / 3);  // safety net, not expected to trigger

    printf("%lld %d %d %lld\n", L, K, Q, W);
    for (auto &kv : mp) printf("%lld %lld\n", kv.first, kv.second);
    return 0;
}
