#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Bidders who want bundles or nothing" (combinatorial
// auction winner determination).
//
// Input:  M N ; c_1..c_M ; then N lines "k p item_1..item_k" (bid i, 1-indexed
//         in input order).
// Output: c ; then c distinct bid ids in [1,N].
//
// Feasibility: for every item j, the number of accepted bids requesting j must
//   not exceed c_j. Any violation, or trailing tokens, scores 0.
//
// Baseline B (checker-computed, order-of-arrival "trivial feasible" reference):
//   scan bids in the INPUT order; accept a bid iff every one of its items still
//   has remaining capacity, decrementing on acceptance. No sorting, no
//   lookahead -- exactly what the "trivial" reference solution reproduces
//   (-> ratio 0.1).
// Score (max): sc = min(1000, 100 * F / max(1,B)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int M = inf.readInt();
    int N = inf.readInt();
    vector<ll> cap(M + 1);
    for (int j = 1; j <= M; j++) cap[j] = inf.readLong();

    vector<int> bk(N + 1);
    vector<ll> bp(N + 1);
    vector<vector<int>> bitems(N + 1);
    for (int i = 1; i <= N; i++) {
        int k = inf.readInt();
        ll p = inf.readLong();
        bk[i] = k;
        bp[i] = p;
        bitems[i].resize(k);
        for (int t = 0; t < k; t++) bitems[i][t] = inf.readInt();
    }

    // ---- internal baseline B: order-of-arrival greedy accept ----
    {
        vector<ll> rem = cap;
        ll B = 0;
        for (int i = 1; i <= N; i++) {
            bool ok = true;
            for (int it : bitems[i]) if (rem[it] < 1) { ok = false; break; }
            if (ok) {
                for (int it : bitems[i]) rem[it]--;
                B += bp[i];
            }
        }
        if (B <= 0) B = 1; // defensive; generator guarantees the first bid always fits

        // ---- replay participant output ----
        int c = ouf.readInt(0, N, "count");
        vector<char> chosen(N + 1, 0);
        vector<ll> used(M + 1, 0);
        ll F = 0;
        for (int t = 0; t < c; t++) {
            int id = ouf.readInt(1, N, "bid id");
            if (chosen[id]) quitf(_wa, "bid %d claimed more than once", id);
            chosen[id] = 1;
            for (int it : bitems[id]) {
                used[it]++;
                if (used[it] > cap[it])
                    quitf(_wa, "item %d oversubscribed: used %lld > supply %lld", it, used[it], cap[it]);
            }
            F += bp[id];
        }
        if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after the id list");

        double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
        quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    }
    return 0;
}
