#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Checker / scorer for debruijn-forbidden-cover (maximize distinct allowed k-mers).
//
// Feasibility of participant string S:
//   * chars all in the alphabet '0'..'0'+A-1;
//   * K <= |S| <= Lmax;
//   * NO length-K window of S is a forbidden factor.
// Objective F = number of DISTINCT allowed length-K factors of S (edges covered).
//
// Internal baseline B = coverage of the NAIVE greedy walk that the checker builds
// itself: start at the lexicographically smallest vertex that has an allowed
// out-edge, repeatedly append the smallest symbol whose edge is allowed and not
// yet used, stop when stuck.  B > 0 (the start vertex has an allowed out-edge).
//
// Score (maximization, affine): Ratio = clamp(0.1 + 0.8*(F-B)/(T-B), 0, 1), where
// T = a^k - |F| is the total number of allowed factors.
//   trivial (== the naive walk)        -> F = B  -> Ratio 0.1
//   max-edge walk on the largest SCC   -> F >> B -> Ratio toward 0.9 (ceiling open).

static int A, K, V, E;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    A = inf.readInt();
    K = inf.readInt();
    int F_cnt = inf.readInt();

    V = 1; for (int i = 0; i < K - 1; i++) V *= A;
    E = V * A;

    // powers of A
    vector<ll> pw(K + 1, 1);
    for (int i = 1; i <= K; i++) pw[i] = pw[i - 1] * A;

    // forbidden edge bitset
    vector<char> forb(E, 0);
    for (int t = 0; t < F_cnt; t++) {
        string tok = inf.readToken();
        // decode length-K string over '0'.. into an edge code
        ll code = 0;
        for (int i = 0; i < K; i++) code = code * A + (tok[i] - '0');
        forb[code] = 1;
    }

    // ---- internal baseline B: naive lexicographic greedy walk ----
    // start = smallest vertex with an allowed out-edge
    int start = -1;
    for (int v = 0; v < V && start < 0; v++)
        for (int s = 0; s < A; s++)
            if (!forb[(ll)v * A + s]) { start = v; break; }
    if (start < 0) quitf(_fail, "bad instance: no allowed edge exists");

    ll B = 0;
    {
        vector<char> used(E, 0);
        int cur = start;
        while (true) {
            int chosen = -1;
            for (int s = 0; s < A; s++) {
                ll ec = (ll)cur * A + s;
                if (!forb[ec] && !used[ec]) { chosen = s; break; }
            }
            if (chosen < 0) break;
            ll ec = (ll)cur * A + chosen;
            used[ec] = 1; B++;
            cur = (int)(ec % V);
        }
    }
    if (B <= 0) quitf(_fail, "bad instance: baseline B=%lld", B);

    // feasibility length bound (generous: room to cover every allowed edge with
    // repeated connector edges plus a spelled-out start prefix)
    ll Lmax = 6LL * E + 2LL * K + 10;

    // ---- read participant string ----
    string S = ouf.readToken();
    if (!ouf.seekEof()) quitf(_wa, "trailing output after the string");

    ll L = (ll)S.size();
    if (L < K) quitf(_wa, "string length %lld < K=%d (needs at least one factor)", L, K);
    if (L > Lmax) quitf(_wa, "string length %lld exceeds Lmax=%lld", L, Lmax);
    for (ll i = 0; i < L; i++) {
        int d = S[i] - '0';
        if (d < 0 || d >= A)
            quitf(_wa, "character '%c' at position %lld is outside the alphabet 0..%c",
                  S[i], i, (char)('0' + A - 1));
    }

    // ---- scan windows: reject forbidden factors, count distinct allowed ----
    // rolling code of the current length-K window
    vector<char> covered(E, 0);
    ll code = 0;
    for (int i = 0; i < K; i++) code = code * A + (S[i] - '0');
    ll F = 0;
    // process first window
    {
        if (forb[code]) quitf(_wa, "string contains a forbidden factor at position 0");
        if (!covered[code]) { covered[code] = 1; F++; }
    }
    ll high = pw[K - 1];  // weight of the leading symbol
    for (ll i = K; i < L; i++) {
        int drop = S[i - K] - '0';
        int add = S[i] - '0';
        code = (code - (ll)drop * high) * A + add;
        if (forb[code])
            quitf(_wa, "string contains a forbidden factor ending at position %lld", i);
        if (!covered[code]) { covered[code] = 1; F++; }
    }

    // ---- affine score anchored between the naive baseline B and the total
    //      allowed edge count T = E - |F|.  Covering the naive walk (F=B) scores
    //      0.1; covering EVERY allowed edge (F=T, only possible if the whole
    //      allowed graph were a single walk-coverable trail) approaches 0.9, so
    //      the reference ceiling stays open and no single walk can saturate. ----
    ll T = E - (ll)F_cnt;                 // total allowed edges
    if (T < B) T = B;                     // safety (should not happen)
    double denom = (double)max((ll)1, T - B);
    double ratio = 0.1 + 0.8 * (double)(F - B) / denom;
    if (ratio < 0.0) ratio = 0.0;
    if (ratio > 1.0) ratio = 1.0;
    quitp(ratio, "OK F=%lld B=%lld T=%lld Ratio: %.6f", F, B, T, ratio);
    return 0;
}
