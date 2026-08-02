#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// chk.cc -- scorer for "Bidders Who Agreed Beforehand" (MAXIMIZE).
//
// Ground truth (exact function of the PUBLIC bids, no hidden channel): a bidder's implied
// rate in auction a is floor(bid/s_a); a bidder is a TRUE RING MEMBER iff that implied rate
// takes more than one distinct value across its market's auctions. Every market plants at
// most one ring, so "true pairs" = all unordered pairs within a market's true set.
//
// Score = 0.5*(F1 over flagged bidders, pooled across markets) + 0.5*(F1 over claimed
// same-group pairs, pooled across markets), compared against the checker's own reference
// construction B (per market: flag only the single highest-total-bid bidder, ties -> smaller
// index, no grouping). ratio = min(1000, 100*F/max(B,eps))/1000.

static double f1(ll tp, ll fp, ll fn) {
    ll denom = 2 * tp + fp + fn;
    if (denom <= 0) return 1.0; // nothing true, nothing claimed -> vacuously correct
    return (2.0 * (double)tp) / (double)denom;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int T = inf.readInt();
    if (T < 1 || T > 100000) quitf(_fail, "generator/input malformed: T=%d", T);

    ll TPm = 0, FPm = 0, FNm = 0, TPp = 0, FPp = 0, FNp = 0;       // participant, pooled
    ll TPmB = 0, FPmB = 0, FNmB = 0, TPpB = 0, FPpB = 0, FNpB = 0; // reference baseline, pooled

    for (int t = 0; t < T; t++) {
        int n = inf.readInt();
        int m = inf.readInt();
        if (n < 1 || n > 5000 || m < 1 || m > 5000)
            quitf(_fail, "generator/input malformed: market %d n=%d m=%d", t, n, m);

        vector<ll> s(m);
        for (int a = 0; a < m; a++) s[a] = inf.readInt(1, 1000000, "s");

        vector<vector<ll>> bid(n, vector<ll>(m));
        vector<ll> totalBid(n, 0);
        for (int i = 0; i < n; i++) {
            for (int a = 0; a < m; a++) {
                bid[i][a] = inf.readInt(1, 100000, "bid");
                totalBid[i] += bid[i][a];
            }
        }

        // ---- ground truth: implied-rate distinctness per bidder ----
        vector<char> isTrue(n, 0);
        for (int i = 0; i < n; i++) {
            set<ll> rates;
            for (int a = 0; a < m; a++) rates.insert(bid[i][a] / s[a]);
            if (rates.size() > 1) isTrue[i] = 1;
        }
        vector<int> trueSet;
        for (int i = 0; i < n; i++) if (isTrue[i]) trueSet.push_back(i);

        auto accumulateSet = [&](const vector<char>& flagged, ll& TP, ll& FP, ll& FN) {
            for (int i = 0; i < n; i++) {
                if (flagged[i] && isTrue[i]) TP++;
                else if (flagged[i] && !isTrue[i]) FP++;
                else if (!flagged[i] && isTrue[i]) FN++;
            }
        };
        auto accumulatePairs = [&](const vector<int>& groupOf, ll& TP, ll& FP, ll& FN) {
            // groupOf[i] = claimed group id (>=0) or -1 if unflagged/no group co-membership
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    bool trueP = isTrue[i] && isTrue[j];
                    bool claimedP = (groupOf[i] >= 0 && groupOf[i] == groupOf[j]);
                    if (claimedP && trueP) TP++;
                    else if (claimedP && !trueP) FP++;
                    else if (!claimedP && trueP) FN++;
                }
            }
        };

        // ---- reference baseline B: the K highest-total-bid bidders, grouped as ONE ring ----
        {
            int K = min(n, 3);
            vector<int> idx(n);
            for (int i = 0; i < n; i++) idx[i] = i;
            sort(idx.begin(), idx.end(), [&](int a, int b) {
                if (totalBid[a] != totalBid[b]) return totalBid[a] > totalBid[b];
                return a < b;
            });
            vector<char> flagged(n, 0);
            vector<int> groupOf(n, -1);
            for (int r = 0; r < K; r++) { flagged[idx[r]] = 1; groupOf[idx[r]] = 0; }
            accumulateSet(flagged, TPmB, FPmB, FNmB);
            accumulatePairs(groupOf, TPpB, FPpB, FNpB);
        }

        // ---- participant output for this market ----
        int g = ouf.readInt(0, n, "g");
        vector<int> groupOf(n, -1);
        vector<char> flagged(n, 0);
        for (int gi = 0; gi < g; gi++) {
            int sz = ouf.readInt(1, n, "sz");
            for (int k = 0; k < sz; k++) {
                int idx = ouf.readInt(0, n - 1, "idx");
                if (groupOf[idx] != -1)
                    quitf(_wa, "market %d: bidder %d listed in more than one group", t, idx);
                groupOf[idx] = gi;
                flagged[idx] = 1;
            }
        }
        accumulateSet(flagged, TPm, FPm, FNm);
        accumulatePairs(groupOf, TPp, FPp, FNp);
    }

    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    double F1mem = f1(TPm, FPm, FNm);
    double F1pair = f1(TPp, FPp, FNp);
    double F = 0.5 * (F1mem + F1pair);

    double F1memB = f1(TPmB, FPmB, FNmB);
    double F1pairB = f1(TPpB, FPpB, FNpB);
    double B = 0.5 * (F1memB + F1pairB);
    if (B <= 0.0) quitf(_fail, "generator/input malformed: baseline B<=0");

    double sc = min(1000.0, 100.0 * F / max(B, 1e-9));
    quitp(sc / 1000.0, "OK F=%.6f B=%.6f F1mem=%.4f F1pair=%.4f Ratio: %.6f",
          F, B, F1mem, F1pair, sc / 1000.0);
    return 0;
}
