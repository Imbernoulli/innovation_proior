#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Journeymen, Masters, and Sweeteners" (near-stable-welfare-forge).
//
// Input: n m e lambda ; then e lines  i j a b s
//   i in [1,n] journeyman, j in [1,m] master
//   a = journeyman i's utility for master j, b = master j's utility for journeyman i
//   s = sweetener (side-payment) cap for pair (i,j)
//
// Output: k ; then k lines  i j p
//   a partial matching (each i, each j used at most once) restricted to given edges,
//   plus an integer transfer p on that pair with -s<=p<=s AND a+p>=0 AND b-p>=0
//   (a payment cannot make either side's realized utility negative).
//   Realized utilities: u_i = a+p (0 if i unmatched), v_j = b-p (0 if j unmatched).
//
// Blocking: an edge (i,j) NOT in the matching blocks iff a > u_i AND b > v_j (both
//   sides would gain by defecting to each other, using their raw edge utility since
//   no payment is in force on a hypothetical pair).
//
// Objective (MAX): F = max(1, sum_{matched}(a+b) - lambda * #blocking ).
//   (Payments are pure transfers -- they cancel out of total welfare -- but they can
//   change WHICH pairs block.)
//
// Baseline B (checker-computed): the "any feasible" greedy-by-input-order matching
//   (scan edges as given, take the pair if both endpoints are still free), zero
//   payments -- scored with the identical formula. This is exactly what
//   solutions/trivial.cpp reproduces (-> ratio 0.1).
// Score: sc = min(1000, 100*F/max(1,B)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();
    int e = inf.readInt();
    ll lambda = inf.readLong();

    vector<int> ei(e), ej(e), ea(e), eb(e), es(e);
    unordered_map<ll, int> edgeIndex;
    edgeIndex.reserve(e * 2 + 8);
    auto key = [&](ll i, ll j) { return i * 4000000LL + j; };
    for (int k = 0; k < e; k++) {
        ei[k] = inf.readInt(1, n, "edge_i");
        ej[k] = inf.readInt(1, m, "edge_j");
        ea[k] = inf.readInt(1, 1000000, "edge_a");
        eb[k] = inf.readInt(1, 1000000, "edge_b");
        es[k] = inf.readInt(0, 1000000, "edge_s");
        edgeIndex[key(ei[k], ej[k])] = k;
    }

    // A tiny helper that, given which edges are matched and the resulting per-side
    // realized utilities, computes (welfare, blocking count) over the WHOLE edge set.
    auto scoreAssignment = [&](const vector<char>& isMatchedEdge, const vector<ll>& u, const vector<ll>& v) -> pair<ll, ll> {
        ll welfare = 0;
        for (int k = 0; k < e; k++) if (isMatchedEdge[k]) welfare += ea[k] + eb[k];
        ll blocking = 0;
        for (int k = 0; k < e; k++) {
            if (isMatchedEdge[k]) continue;
            if (ea[k] > u[ei[k]] && eb[k] > v[ej[k]]) blocking++;
        }
        return {welfare, blocking};
    };

    // ---- internal baseline B: greedy-by-input-order matching, zero payments ----
    ll B;
    {
        vector<char> usedI(n + 1, 0), usedJ(m + 1, 0), isMatched(e, 0);
        vector<ll> u(n + 1, 0), v(m + 1, 0);
        for (int k = 0; k < e; k++) {
            if (!usedI[ei[k]] && !usedJ[ej[k]]) {
                usedI[ei[k]] = usedJ[ej[k]] = 1;
                isMatched[k] = 1;
                u[ei[k]] = ea[k];
                v[ej[k]] = eb[k];
            }
        }
        auto pr = scoreAssignment(isMatched, u, v);
        ll Fbase = pr.first - lambda * pr.second;
        if (Fbase < 1) Fbase = 1;   // same floor as the participant F below, so the
        B = Fbase;                 // baseline's own construction always replays to ratio 0.1
    }

    // ---- replay participant's matching + payments ----
    int k = ouf.readInt(0, min(n, m), "k");
    vector<char> usedI(n + 1, 0), usedJ(m + 1, 0), isMatched(e, 0);
    vector<ll> u(n + 1, 0), v(m + 1, 0);
    for (int t = 0; t < k; t++) {
        int pi = ouf.readInt(1, n, "i");
        int pj = ouf.readInt(1, m, "j");
        auto it = edgeIndex.find(key(pi, pj));
        if (it == edgeIndex.end())
            quitf(_wa, "pair (%d,%d) is not an edge of the input market", pi, pj);
        if (usedI[pi]) quitf(_wa, "journeyman %d matched more than once", pi);
        if (usedJ[pj]) quitf(_wa, "master %d matched more than once", pj);
        int idx = it->second;
        ll a = ea[idx], b = eb[idx], s = es[idx];
        ll lo = max(-s, -a), hi = min(s, b); // keeps a+p>=0 and b-p>=0 automatically
        ll p = ouf.readLong(lo, hi, "payment");
        usedI[pi] = usedJ[pj] = 1;
        isMatched[idx] = 1;
        u[pi] = a + p;
        v[pj] = b - p;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    auto pr = scoreAssignment(isMatched, u, v);
    ll F = pr.first - lambda * pr.second;
    if (F < 1) F = 1;   // same floor as B: a feasible-but-blocking-drowned matching is as bad
                        // as it gets, not "worse than the baseline's own floor"

    double sc = min(1000.0, 100.0 * (double)F / (double)B);
    quitp(sc / 1000.0, "OK F=%lld B=%lld blocking=%lld Ratio: %.6f", F, B, pr.second, sc / 1000.0);
    return 0;
}
