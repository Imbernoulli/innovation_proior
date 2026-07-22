#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef unsigned long long ull;

// -----------------------------------------------------------------------------
// Checker / scorer for "Grow a Sharp Crystal, Not a Blob".
//
// Input:  SEED K M ; then K integers r_0..r_{K-1} (target star-polygon vertices,
//         vertex i at angle 2*pi*i/K, distance r_i).
// Output: 72 nonnegative reals w[0..71] (relative growth BUDGET over 5-degree bins).
//
// The checker converts w[] into an exact per-bin cell QUOTA via largest-remainder
// apportionment of M-1 cells (the seed itself is free, always "grown"), then runs
// a 4-neighbor Eden process from the seed: at each step, a candidate frontier
// cell is drawn uniformly at random; if its direction-bin's quota is already
// full it is discarded (never revisited), otherwise it is grown and its bin's
// count increments. This makes a direction's FINAL reach a deterministic
// function of its declared quota (a hard, self-limiting budget -- no direction
// can "run away"), so matching the target silhouette is entirely about getting
// the 72 quotas right. All randomness is drawn from SEED alone (never from the
// participant's output), so the same w[] always regrows the same cluster.
//
// F = |grown(w) intersect Target|.  B = |grown(uniform w=1) intersect Target|
// (an isotropic quota split -- every bin gets an equal budget regardless of the
// target's actual shape: the KPZ-universal round blob). Both grown clusters and
// Target all have exactly M cells by construction, so
//   ratio = min(1000, 100*F/max(1,B)) / 1000
// makes the isotropic split score exactly 0.100, any anisotropy that raises the
// overlap scores higher (capped at 1.0).
// -----------------------------------------------------------------------------

static const int KBINS = 72;
static const double TWO_PI = 2.0 * M_PI;

static double boundaryRadius(int K, const vector<double>& r, double theta) {
    double sector = TWO_PI / K;
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

struct RNG {
    ull s;
    RNG(ull seed) { s = seed ? seed : 88172645463325252ULL; }
    ull nxt() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    double nextDouble() { return (double)(nxt() >> 11) * (1.0 / 9007199254740992.0); }
};

static int binOfAngle(double theta) {
    if (theta < 0) theta += TWO_PI;
    int b = (int)floor(theta * KBINS / TWO_PI);
    if (b < 0) b = 0;
    if (b >= KBINS) b = KBINS - 1;
    return b;
}

// Apportion `total` indivisible units across KBINS bins in proportion to w[],
// via the largest-remainder method (Hamilton apportionment): deterministic,
// sums to exactly `total`.
static array<ll, KBINS> apportion(const array<double, KBINS>& w, ll total) {
    array<ll, KBINS> quota{};
    double wsum = 0;
    for (double v : w) wsum += v;
    if (!(wsum > 0.0) || total <= 0) {
        if (total > 0) quota[0] = total;
        return quota;
    }
    array<double, KBINS> raw{};
    ll used = 0;
    vector<pair<double, int>> fracs(KBINS);
    for (int i = 0; i < KBINS; i++) {
        raw[i] = w[i] / wsum * (double)total;
        quota[i] = (ll)floor(raw[i]);
        used += quota[i];
        fracs[i] = {raw[i] - (double)quota[i], i};
    }
    ll rem = total - used;
    // Deterministic tie-break by bin index: std::sort is not guaranteed stable,
    // so equal-fraction bins must be ordered explicitly (never left to
    // implementation-defined comparator behavior) for the apportionment to be
    // reproducible across compilers/platforms, not just across runs.
    sort(fracs.begin(), fracs.end(), [](const pair<double, int>& a, const pair<double, int>& b) {
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    });
    for (ll k = 0; k < rem; k++) quota[fracs[(size_t)(k % KBINS)].second] += 1;
    return quota;
}

static ll growAndScore(ull seed, int K, const vector<double>& r, ll M,
                        const array<double, KBINS>& w) {
    RNG rng(seed);
    array<ll, KBINS> quota = apportion(w, M - 1);
    array<ll, KBINS> cnt{};

    static const ll ddx[4] = {1, -1, 0, 0}, ddy[4] = {0, 0, 1, -1};
    auto key = [](ll dx, ll dy) -> ll { return (dx + 3000000LL) * 6000001LL + (dy + 3000000LL); };

    unordered_set<ll> occupiedSet;
    unordered_set<ll> inFront;
    vector<pair<ll, ll>> occList;
    vector<pair<ll, ll>> frontier;

    occupiedSet.insert(key(0, 0));
    occList.push_back({0, 0});
    for (int d = 0; d < 4; d++) {
        ll nx = ddx[d], ny = ddy[d];
        inFront.insert(key(nx, ny));
        frontier.push_back({nx, ny});
    }
    ll occCount = 1;
    ll guard = 0;
    ll maxGuard = 200LL * M + 200000LL;

    while (occCount < M && !frontier.empty() && guard < maxGuard) {
        guard++;
        int sz = (int)frontier.size();
        int idx = (int)(rng.nextDouble() * sz);
        if (idx >= sz) idx = sz - 1;
        ll pdx = frontier[idx].first, pdy = frontier[idx].second;
        double theta = atan2((double)pdy, (double)pdx);
        int b = binOfAngle(theta);
        if (cnt[b] >= quota[b]) {
            // this bin's budget is spent: discard permanently, no growth step.
            frontier[idx] = frontier[sz - 1];
            frontier.pop_back();
            inFront.erase(key(pdx, pdy));
            continue;
        }
        frontier[idx] = frontier[sz - 1];
        frontier.pop_back();
        ll pk = key(pdx, pdy);
        inFront.erase(pk);
        occupiedSet.insert(pk);
        occList.push_back({pdx, pdy});
        cnt[b]++;
        occCount++;
        for (int d = 0; d < 4; d++) {
            ll nx = pdx + ddx[d], ny = pdy + ddy[d];
            ll kk = key(nx, ny);
            if (!occupiedSet.count(kk) && !inFront.count(kk)) {
                inFront.insert(kk);
                frontier.push_back({nx, ny});
            }
        }
    }
    // Safety net: a heavily concentrated field can gate off every direction
    // that is actually REACHABLE from the current cluster (e.g. all budget on
    // one bin that the seed's 4 axis-aligned neighbors never fall into), which
    // permanently empties `frontier` while occCount < M. Cells discarded by
    // the quota gate above are only removed from OUR bookkeeping -- they are
    // still empty and still adjacent to the cluster -- so rebuild the frontier
    // from scratch (ignoring quota state) before finishing growth. This
    // guarantees the cluster always reaches EXACTLY mass M, matching the
    // scoring formula's |C|=|Target|=M invariant, regardless of how
    // unreachable the declared field makes some directions.
    if (occCount < M) {
        frontier.clear();
        inFront.clear();
        for (auto& c : occList) {
            for (int d = 0; d < 4; d++) {
                ll nx = c.first + ddx[d], ny = c.second + ddy[d];
                ll kk = key(nx, ny);
                if (!occupiedSet.count(kk) && !inFront.count(kk)) {
                    inFront.insert(kk);
                    frontier.push_back({nx, ny});
                }
            }
        }
    }
    while (occCount < M && !frontier.empty()) {
        int sz = (int)frontier.size();
        int idx = (int)(rng.nextDouble() * sz);
        if (idx >= sz) idx = sz - 1;
        ll pdx = frontier[idx].first, pdy = frontier[idx].second;
        frontier[idx] = frontier[sz - 1];
        frontier.pop_back();
        ll pk = key(pdx, pdy);
        inFront.erase(pk);
        occupiedSet.insert(pk);
        occList.push_back({pdx, pdy});
        occCount++;
        for (int d = 0; d < 4; d++) {
            ll nx = pdx + ddx[d], ny = pdy + ddy[d];
            ll kk = key(nx, ny);
            if (!occupiedSet.count(kk) && !inFront.count(kk)) {
                inFront.insert(kk);
                frontier.push_back({nx, ny});
            }
        }
    }

    ll inter = 0;
    for (auto& c : occList) {
        double dx = (double)c.first, dy = (double)c.second;
        double rho = sqrt(dx * dx + dy * dy);
        if (rho < 1e-9) { inter++; continue; }
        double theta = atan2(dy, dx);
        if (theta < 0) theta += TWO_PI;
        double bnd = boundaryRadius(K, r, theta);
        if (rho <= bnd + 1e-9) inter++;
    }
    return inter;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    ull seed = (ull)inf.readLong(1LL, 2000000000LL, "SEED");
    int K = inf.readInt(4, 16, "K");
    ll M = inf.readLong(1LL, 2000000LL, "M");
    vector<double> r(K);
    for (int i = 0; i < K; i++) r[i] = (double)inf.readLong(1LL, 200LL, "r_i");

    if (M < 1) quitf(_fail, "generator produced non-positive M");

    array<double, KBINS> w{};
    double wsum = 0;
    for (int i = 0; i < KBINS; i++) {
        double v = ouf.readDouble(0.0, 1e12, "w_i");
        if (!isfinite(v)) quitf(_wa, "w[%d] is not finite", i);
        w[i] = v;
        wsum += v;
    }
    if (!(wsum > 0.0)) quitf(_wa, "all growth-budget weights are zero");
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after the 72 weights");

    array<double, KBINS> uni{};
    for (int i = 0; i < KBINS; i++) uni[i] = 1.0;

    ll F = growAndScore(seed, K, r, M, w);
    ll B = growAndScore(seed, K, r, M, uni);
    if (B <= 0) B = 1;

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld M=%lld Ratio: %.6f", F, B, M, sc / 1000.0);
    return 0;
}
