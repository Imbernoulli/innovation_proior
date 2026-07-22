// chk.cc -- checker/scorer for "Slitting Coils When Knives Crawl, Not Jump"
//
// Validates a knife-motion-bounded slitting campaign and scores it against an
// internal naive baseline (one-unit-at-a-time, input-order, no batching).
#include "testlib.h"
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
using namespace std;
typedef long long ll;

static ll gW; static int gK;

// Build the K-knife pattern that repeats `width` as many times as usefully fits
// (capped at K copies), padding the rest with 1-unit filler + a remainder segment.
// Mirrors the construction used by the trivial/greedy/strong reference solutions.
static pair<vector<ll>, ll> fullPattern(ll W, int K, ll width) {
    ll c = min((ll)K, (W - 1) / width);
    while (c >= 0) {
        ll remSlots = K + 1 - c;
        if (remSlots <= 0) { c--; continue; }
        ll fillerCount = remSlots - 1;
        ll remainder = W - c * width - fillerCount * 1;
        if (remainder >= 1) {
            vector<ll> segs;
            for (ll k = 0; k < c; k++) segs.push_back(width);
            for (ll k = 0; k < fillerCount; k++) segs.push_back(1);
            segs.push_back(remainder);
            vector<ll> pos; ll run = 0;
            for (size_t k = 0; k + 1 < segs.size(); k++) { run += segs[k]; pos.push_back(run); }
            return {pos, c};
        }
        c--;
    }
    vector<ll> pos;
    ll run = 0;
    for (int k = 0; k < K; k++) { run += max((ll)1, (W - 1) / (K + 1)); pos.push_back(min(run, W - 1)); }
    return {pos, 0};
}

static ll hopCost(const vector<ll> &a, const vector<ll> &b, ll m) {
    ll best = 0;
    for (size_t j = 0; j < a.size(); j++) {
        ll d = llabs(a[j] - b[j]);
        ll h = (d + m - 1) / m;
        best = max(best, h);
    }
    return best;
}

// Bridge from `cur` to `target`, one motion-bound-respecting hop at a time. Each hop
// moves every knife toward its target by a fraction chosen so the largest remaining
// gap shrinks by exactly m; the fraction is recomputed from the CURRENT position each
// hop (not from a fixed total-step count), so the very last hop always lands exactly
// on target the moment the remaining gap is safely <= m (no rounding risk there).
// Intermediate hops are clamped to [-m, +m] per knife (belt-and-suspenders against
// rounding) and the strictly-increasing order is repaired left-to-right. Returns the
// list of intermediate+final knife-position vectors (empty if already at target).
static vector<vector<ll>> bridgeTo(vector<ll> cur, const vector<ll> &target, ll m, int K) {
    vector<vector<ll>> path;
    vector<ll> p = cur;
    int guard = 0;
    while (true) {
        ll maxd = 0;
        for (int j = 0; j < K; j++) maxd = max(maxd, llabs(target[j] - p[j]));
        if (maxd == 0) break;
        vector<ll> rp(K);
        if (maxd <= m) {
            rp = target;
        } else {
            double f = (double)m / (double)maxd;
            for (int j = 0; j < K; j++) {
                double v = p[j] + (target[j] - p[j]) * f;
                ll iv = (ll)llround(v);
                if (iv > p[j] + m) iv = p[j] + m;
                if (iv < p[j] - m) iv = p[j] - m;
                rp[j] = iv;
            }
            for (int j = 1; j < K; j++) if (rp[j] <= rp[j - 1]) rp[j] = rp[j - 1] + 1;
            for (int j = 0; j < K; j++) {
                if (rp[j] > p[j] + m) rp[j] = p[j] + m;
                if (rp[j] < p[j] - m) rp[j] = p[j] - m;
            }
            for (int j = 1; j < K; j++) if (rp[j] <= rp[j - 1]) rp[j] = rp[j - 1] + 1;
        }
        path.push_back(rp);
        p = rp;
        if (++guard > 200000) break;   // safety valve; never triggered in practice
    }
    return path;
}

// The checker's OWN trivial construction: visit each demand type once, in the
// given input order, using the single-type pattern above -- but with NO batching
// insight (run length 1, repeated qty_i times) and NO reachability planning
// (just bridge straight to the next type in input order). This is the same
// strategy solutions/trivial.cpp implements; recomputing it here (rather than
// trusting the submission) keeps the baseline honest and deterministic.
static ll internalBaselineB(ll W, int K, ll m, vector<ll> S,
                             const vector<ll> &width, const vector<ll> &qty,
                             ll maxSetups, ll setupCost, ll penalty) {
    int n = (int)width.size();
    vector<ll> cur = S;
    ll material = 0, setups = 0;
    vector<ll> produced(n, 0);
    for (int i = 0; i < n; i++) {
        auto pr = fullPattern(W, K, width[i]);
        vector<ll> pos = pr.first;
        auto path = bridgeTo(cur, pos, m, K);
        for (size_t s = 0; s < path.size() && setups < maxSetups; s++) {
            setups++; material += W; cur = path[s];
        }
        if (setups >= maxSetups) break;
        for (ll u = 0; u < qty[i] && setups < maxSetups; u++) {
            setups++; material += W; produced[i]++;
        }
    }
    ll shortfall = 0;
    for (int i = 0; i < n; i++) if (produced[i] < qty[i]) shortfall += (qty[i] - produced[i]) * width[i];
    return material + setupCost * setups + penalty * shortfall;
}

int main(int argc, char *argv[]) {
    registerTestlibCmd(argc, argv);

    ll W = inf.readLong(30LL, 3000LL, "W");
    int K = inf.readInt(2, 8, "K");
    ll m = inf.readLong(1LL, W, "m");
    gW = W; gK = K;
    vector<ll> S(K);
    for (int j = 0; j < K; j++) S[j] = inf.readLong(1LL, W - 1, "S_j");

    int n = inf.readInt(1, 30, "n");
    vector<ll> width(n), qty(n);
    map<ll, int> widthIndex;
    for (int i = 0; i < n; i++) {
        // Cap at W-K-1: fullPattern() needs at least 1 real copy of width_i plus
        // K-1 filler segments (>=1 each) to fit inside W, or it silently degrades to
        // a c=0 pattern that can never actually contain width_i (see fullPattern's
        // fallback branch below). Keeping every demanded width inside this cap
        // guarantees the checker's own baseline construction (and the trivial/
        // greedy/strong references) can always genuinely produce it.
        ll wcap = max((ll)1, W - (ll)K - 1);
        width[i] = inf.readLong(1LL, wcap, "width_i");
        qty[i] = inf.readLong(1LL, 1000000LL, "qty_i");
        widthIndex[width[i]] = i;
    }
    ll maxSetups = inf.readLong(1LL, 100000LL, "maxSetups");
    ll setupCost = inf.readLong(1LL, 10000000LL, "setupCost");
    ll penalty = inf.readLong(1LL, 100000LL, "penalty");

    ll T = ouf.readLong(1LL, maxSetups, "T");
    vector<ll> prevPos = S;
    ll totalMaterial = 0;
    vector<ll> produced(n, 0);

    for (ll t = 0; t < T; t++) {
        vector<ll> pos(K);
        for (int j = 0; j < K; j++) pos[j] = ouf.readLong(1LL, W - 1, "p_j");
        for (int j = 1; j < K; j++)
            if (pos[j] <= pos[j - 1])
                quitf(_wa, "setup %lld: knife positions not strictly increasing (p_%d=%lld <= p_%d=%lld)",
                      t + 1, j + 1, pos[j], j, pos[j - 1]);
        ll r = ouf.readLong(1LL, (ll)2e9, "r_t");
        for (int j = 0; j < K; j++) {
            ll d = llabs(pos[j] - prevPos[j]);
            if (d > m)
                quitf(_wa, "setup %lld: knife %d moved %lld > bound m=%lld", t + 1, j + 1, d, m);
        }
        prevPos = pos;
        totalMaterial += r * W;
        ll prevc = 0;
        for (int j = 0; j < K; j++) {
            ll seg = pos[j] - prevc; prevc = pos[j];
            auto it = widthIndex.find(seg);
            if (it != widthIndex.end()) produced[it->second] += r;
        }
        ll lastSeg = W - prevc;
        auto it = widthIndex.find(lastSeg);
        if (it != widthIndex.end()) produced[it->second] += r;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens after the campaign");

    ll shortfall = 0;
    for (int i = 0; i < n; i++)
        if (produced[i] < qty[i]) shortfall += (qty[i] - produced[i]) * width[i];

    ll F = totalMaterial + setupCost * T + penalty * shortfall;
    ll B = internalBaselineB(W, K, m, S, width, qty, maxSetups, setupCost, penalty);
    if (B <= 0) B = 1;

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
