#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// ---------------------------------------------------------------------------
// Tailor Matching Plaid Across Every Seam -- checker.
// Input:  N W r
//         N lines: w_i h_i req_i
//         M
//         M lines: i e_i j e_j   (seam pairs, edge 0=bottom 1=top)
// Output: N lines: x_i y_i f_i
//
// Feasibility: containment, pairwise non-overlap, each piece's own
// stripe-repeat-congruence under its own f_i, and every seam's two edge
// phases (mod r) equal.
// Objective F = max_i(y_i+h_i) (minimize). Internal baseline B = ceil(area/W).
// Score: ratio = min(1, B/F).
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt(1, 1000, "N");
    int W = inf.readInt(1, 20000, "W");
    long long r = inf.readLong(3LL, 20000LL, "r");

    vector<long long> w(N), h(N), req(N);
    for (int i = 0; i < N; i++) {
        w[i] = inf.readLong(1LL, (long long)W, "w");
        h[i] = inf.readLong(1LL, (long long)200000, "h");
        req[i] = inf.readLong(0LL, r - 1, "req");
    }
    int M = inf.readInt(0, N, "M");
    vector<int> pi(M), ei(M), pj(M), ej(M);
    for (int k = 0; k < M; k++) {
        pi[k] = inf.readInt(0, N - 1, "i");
        ei[k] = inf.readInt(0, 1, "ei");
        pj[k] = inf.readInt(0, N - 1, "j");
        ej[k] = inf.readInt(0, 1, "ej");
    }

    vector<long long> x(N), y(N);
    vector<int> f(N);
    for (int i = 0; i < N; i++) {
        x[i] = ouf.readLong(0, (long long)W, "x");
        if (x[i] + w[i] > (long long)W)
            quitf(_wa, "piece %d: x+w exceeds fabric width", i);
        y[i] = ouf.readLong(0, (long long)2000000000LL, "y");
        f[i] = ouf.readInt(0, 1, "f");
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens after %d pieces", N);

    // pairwise non-overlap (touching allowed)
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            bool xover = x[i] < x[j] + w[j] && x[j] < x[i] + w[i];
            bool yover = y[i] < y[j] + h[j] && y[j] < y[i] + h[i];
            if (xover && yover) quitf(_wa, "pieces %d and %d overlap", i, j);
        }
    }

    auto target = [&](int i, int fi) -> long long {
        long long t;
        if (fi == 0) t = ((-req[i]) % r + r) % r;
        else t = ((req[i] - h[i]) % r + r) % r;
        return t;
    };
    for (int i = 0; i < N; i++) {
        long long need = target(i, f[i]);
        long long ym = ((y[i] % r) + r) % r;
        if (ym != need)
            quitf(_wa, "piece %d violates its own stripe-repeat-congruence (f=%d)", i, f[i]);
    }

    auto phase = [&](int p, int e) -> long long {
        if (e == 0) return ((y[p] % r) + r) % r;
        long long v = (y[p] + h[p]) % r;
        return (v + r) % r;
    };
    for (int k = 0; k < M; k++) {
        if (phase(pi[k], ei[k]) != phase(pj[k], ej[k]))
            quitf(_wa, "seam pair %d phase mismatch", k);
    }

    long long F = 0;
    for (int i = 0; i < N; i++) F = max(F, y[i] + h[i]);
    // internal baseline: the trivial "do-nothing" construction that never
    // shares fabric width at all (every piece serialized alone) -- its
    // length is exactly the sum of piece heights, regardless of rounding.
    long long B = 0;
    for (int i = 0; i < N; i++) B += h[i];
    if (B < 1) B = 1;

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    double ratio = sc / 1000.0;
    quitp(ratio, "OK F=%lld B=%lld Ratio: %.6f", F, B, ratio);
    return 0;
}
