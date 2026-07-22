#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Checker/scorer for parallax-scope-pairing ("Twin-Eye Transient Watch").
//
// Feasibility: strict bounded reads + explicit exact-arrival replay per
// telescope (no idling allowed -- start tick must equal the exact arrival
// tick from the previous commitment / parked start). Any violation -> whole
// output scores 0.
//
// Objective F: for each target, among the observations actually made of it,
// find the earliest tick at which a cross-site pair completes (two
// observations, different sites, start-tick gap <= W) by scanning
// observations of that target in increasing start-tick order and checking a
// trailing window of size W for an opposite-site observation -- the first
// hit is provably the pair with the smallest possible "later" tick, hence
// the least-decayed, highest-value pair. Sum the decayed payout of the
// winning pair (if any) over all targets.
//
// Baseline B: largest v[j] among targets with a[j]=0 for which some site-0
// telescope AND some site-1 telescope are already parked exactly at
// pos[j] at tick 0 (payout at tick 0 = v[j] exactly, no decay). The
// generator always plants at least one such target, so B >= 1 always.

static int T, M, H, W;
static ll Pn, Qd;
static vector<int> siteOf, posOf, speedOf;
static vector<int> aOf, posT, vOf, oOf;

static inline int angdist(int a, int b) {
    int d = abs(a - b) % 360;
    return min(d, 360 - d);
}
static inline int travelTicks(int p, int q, int speed) {
    int d = angdist(p, q);
    if (d == 0) return 0;
    return (d + speed - 1) / speed;
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    T = inf.readInt();
    M = inf.readInt();
    H = inf.readInt();
    W = inf.readInt();
    Pn = inf.readLong();
    Qd = inf.readLong();

    siteOf.resize(T); posOf.resize(T); speedOf.resize(T);
    for (int i = 0; i < T; i++) {
        siteOf[i] = inf.readInt();
        posOf[i] = inf.readInt();
        speedOf[i] = inf.readInt();
    }
    aOf.resize(M); posT.resize(M); vOf.resize(M); oOf.resize(M);
    for (int j = 0; j < M; j++) {
        aOf[j] = inf.readInt();
        posT[j] = inf.readInt();
        vOf[j] = inf.readInt();
        oOf[j] = inf.readInt();
    }

    // ---- internal baseline B: SUM of every zero-travel, zero-decay freebie
    // (a[j]=0 with some site-0 AND some site-1 telescope already parked
    // exactly at pos[j]). The generator plants one such freebie per anchor
    // pair, with the count scaling with instance size -- matching this
    // baseline's own trivial strategy ("grab every free pair") lets B track
    // the scale of the instance instead of pinning it to a single constant,
    // which is what keeps the strong reference from saturating the cap on
    // large tests. ----
    ll B = 0;
    for (int j = 0; j < M; j++) {
        if (aOf[j] != 0) continue;
        bool has0 = false, has1 = false;
        for (int i = 0; i < T; i++) {
            if (posOf[i] == posT[j]) {
                if (siteOf[i] == 0) has0 = true;
                else has1 = true;
            }
        }
        if (has0 && has1) B += vOf[j];
    }
    if (B < 1) B = 1; // guard; generator always plants at least one real freebie

    const int KMAX = H + 5; // generous: min duration 2 caps real visit count well below this

    // observations[j] = list of (startTick, site)
    vector<vector<pair<int,int>>> obs(M);
    ll totalVisits = 0;

    for (int i = 0; i < T; i++) {
        int k = ouf.readInt(0, KMAX, "k_i");
        int prevEnd = 0, prevPos = posOf[i]; // "previous commitment" = the parked start
        for (int l = 0; l < k; l++) {
            int t = ouf.readInt(0, H - 1, "t");
            int j = ouf.readInt(0, M - 1, "j");
            if (t + oOf[j] > H)
                quitf(_wa, "telescope %d visit %d: observation of target %d runs past horizon (t=%d o=%d H=%d)",
                      i, l, j, t, oOf[j], H);
            if (t < aOf[j])
                quitf(_wa, "telescope %d visit %d: target %d observed at t=%d before it appears (a=%d)",
                      i, l, j, t, aOf[j]);
            int need = prevEnd + travelTicks(prevPos, posT[j], speedOf[i]);
            if (t != need)
                quitf(_wa, "telescope %d visit %d: start tick %d != required exact arrival tick %d (no idling allowed)",
                      i, l, t, need);
            obs[j].push_back({t, siteOf[i]});
            prevEnd = t + oOf[j];
            prevPos = posT[j];
            totalVisits++;
            if (totalVisits > 2000000LL)
                quitf(_wa, "output too large");
        }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens after the schedule");

    // ---- objective F ----
    ll F = 0;
    for (int j = 0; j < M; j++) {
        auto &v = obs[j];
        if (v.size() < 2) continue;
        sort(v.begin(), v.end());
        int lo = 0;
        int cnt0 = 0, cnt1 = 0;
        int pairTick = -1;
        for (int i = 0; i < (int)v.size(); i++) {
            while (v[lo].first < v[i].first - W) {
                if (v[lo].second == 0) cnt0--; else cnt1--;
                lo++;
            }
            // window [lo, i-1] already accumulated; check opposite site present
            if (v[i].second == 0) {
                if (cnt1 > 0) { pairTick = v[i].first; break; }
            } else {
                if (cnt0 > 0) { pairTick = v[i].first; break; }
            }
            if (v[i].second == 0) cnt0++; else cnt1++;
        }
        if (pairTick < 0) continue;
        int dt = pairTick - aOf[j];
        ll val = vOf[j];
        for (int d = 0; d < dt; d++) {
            val = (val * Pn) / Qd;
            if (val <= 0) { val = 0; break; }
        }
        F += val;
    }

    double sc = min(1000.0, 100.0 * (double)F / (double)max(1LL, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
