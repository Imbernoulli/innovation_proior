#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef unsigned long long u64;

// Checker/scorer for stuckat-test-compression.
//
// Input:  K N, then N faults, each "c p1 v1 p2 v2 ... pc vc" (a CUBE: fixed
//         (position,value) requirements over the K control lines).
// Output: P, then P probe lines, each a length-K string of '0'/'1'.
//
// Feasibility: every fault's cube must be MATCHED (all its (p,v) pairs
// satisfied) by at least one probe.  1 <= P <= N (never useful to exceed N:
// one dedicated probe per fault always suffices).
//
// Objective F = P (minimize).  Internal baseline B = N (the "one dedicated
// probe per fault" trivial construction -- always feasible, always exactly N).
//   trivial (P=N)                         -> Ratio 0.1
//   exploits dominance + disjoint orbits  -> Ratio much higher (ceiling open,
//                                             the clique block forces P >= m,
//                                             so no submission saturates 1.0).

static int K, N, W;   // W = number of 64-bit words per bit-vector

static inline void setbit(vector<u64> &v, int p) { v[p >> 6] |= (1ULL << (p & 63)); }
static inline bool getbit(const vector<u64> &v, int p) { return (v[p >> 6] >> (p & 63)) & 1ULL; }

// does probe (full assignment) satisfy cube (one,zero)?
static inline bool satisfies(const vector<u64> &probe, const vector<u64> &one, const vector<u64> &zero) {
    for (int w = 0; w < W; w++) {
        u64 pb = probe[w];
        if ((pb & one[w]) != one[w]) return false;
        if ((~pb & zero[w]) != zero[w]) return false;
    }
    return true;
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    K = inf.readInt();
    N = inf.readInt();
    W = (K + 63) / 64;

    vector<vector<u64>> one(N, vector<u64>(W, 0)), zero(N, vector<u64>(W, 0));
    for (int i = 0; i < N; i++) {
        int c = inf.readInt();
        for (int k = 0; k < c; k++) {
            int p = inf.readInt();
            int v = inf.readInt();
            if (v == 1) setbit(one[i], p); else setbit(zero[i], p);
        }
    }

    int P = ouf.readInt(1, N, "P");
    vector<vector<u64>> probes(P, vector<u64>(W, 0));
    for (int j = 0; j < P; j++) {
        string tok = ouf.readToken();
        if ((int)tok.size() != K)
            quitf(_wa, "probe %d has length %d, expected K=%d", j + 1, (int)tok.size(), K);
        for (int p = 0; p < K; p++) {
            char ch = tok[p];
            if (ch == '1') setbit(probes[j], p);
            else if (ch != '0')
                quitf(_wa, "probe %d has non-binary character '%c' at position %d", j + 1, ch, p);
        }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens after the probe list");

    for (int i = 0; i < N; i++) {
        bool detected = false;
        for (int j = 0; j < P; j++)
            if (satisfies(probes[j], one[i], zero[i])) { detected = true; break; }
        if (!detected) quitf(_wa, "fault %d (0-indexed) is not detected by any probe", i);
    }

    ll F = P;
    ll B = N;
    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
