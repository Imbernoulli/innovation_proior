#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// ---------------------------------------------------------------------------
// No-Tell Round Robin -- checker.
//
// Input:  n  (odd, 3 <= n <= 1001)
// Output: n-1 lines; line i (1-indexed, i=1..n-1) is a string of n-i chars
//         over {0,1}. Char k of line i describes team i vs team (i+k):
//         '1' = i beats i+k, '0' = i+k beats i.
//
// F = DegreeSkew + QuadSkew (see statement), computed on the participant's
// schedule. B = same formula computed on the checker's own fully-transitive
// reference schedule (i beats j iff i<j). Score is a fixed smooth increasing
// function of B/F that is exactly 0.1 at F=B, never exceeds 1.0, and
// approaches (but never reaches) 1.0 as F shrinks far below B.
// ---------------------------------------------------------------------------

static long long computeDefect(int n, vector<vector<uint64_t>>& A, int W) {
    // out-degree via popcount
    vector<long long> outdeg(n, 0);
    for (int i = 0; i < n; i++) {
        long long c = 0;
        for (int w = 0; w < W; w++) c += __builtin_popcountll(A[i][w]);
        outdeg[i] = c;
    }
    long long Ddeg = 0;
    for (int i = 0; i < n; i++) {
        long long d = 2 * outdeg[i] - (long long)(n - 1);
        Ddeg += d * d;
    }
    long long D4 = 0;
    for (int u = 0; u < n; u++) {
        for (int v = u + 1; v < n; v++) {
            long long c = 0;
            for (int w = 0; w < W; w++) c += __builtin_popcountll(A[u][w] & A[v][w]);
            long long dev = 4 * c - (long long)(n - 2);
            D4 += dev * dev;
        }
    }
    return Ddeg + D4;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt(3, 1001, "n");

    int W = (n + 63) / 64;
    vector<vector<uint64_t>> A(n, vector<uint64_t>(W, 0));

    for (int i = 0; i < n - 1; i++) {
        string tok = ouf.readToken();
        int expected = n - 1 - i;
        if ((int)tok.size() != expected) {
            quitf(_wa, "row %d: expected length %d, got %d", i + 1, expected, (int)tok.size());
        }
        for (int k = 0; k < expected; k++) {
            char c = tok[k];
            if (c != '0' && c != '1') {
                quitf(_wa, "row %d has invalid character '%c' at position %d (expected '0' or '1')", i + 1, c, k + 1);
            }
            int j = i + 1 + k; // 0-indexed opponent
            if (c == '1') {
                A[i][j >> 6] |= (1ULL << (j & 63));
            } else {
                A[j][i >> 6] |= (1ULL << (i & 63));
            }
        }
    }
    if (!ouf.seekEof()) {
        quitf(_wa, "trailing output after %d rows", n - 1);
    }

    long long F = computeDefect(n, A, W);

    // Internal reference baseline: fully transitive schedule, i beats j iff i<j.
    vector<vector<uint64_t>> R(n, vector<uint64_t>(W, 0));
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            R[i][j >> 6] |= (1ULL << (j & 63));
        }
    }
    long long B = computeDefect(n, R, W);
    if (B <= 0) B = 1;

    double Fd = (double)max(1LL, F);
    double ratio = (double)B / Fd;
    double lr = log(max(ratio, 1e-9));
    double s;
    if (lr <= 0.0) {
        s = 0.1 * ratio;
    } else {
        s = 0.1 + 0.82 * lr / (lr + 1.5);
    }
    if (s < 0.0) s = 0.0;
    if (s > 1.0) s = 1.0;

    quitp(s, "OK F=%lld B=%lld Ratio: %.6f", F, B, s);
    return 0;
}
