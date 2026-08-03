#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// chk.cc -- scorer for greenhouse climate-program assignment.
// Minimization. Baseline B = conflict of the all-same-program assignment = sum of all
// coupling strengths (positive). Participant conflict F = sum of strengths of coupled
// pairs whose two zones share a program.  ratio = min(1, (B / max(1,F)) / 10).
int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int M = inf.readInt();
    int K = inf.readInt();

    vector<int> eu(M), ev(M);
    vector<long long> ew(M);
    long long B = 0;               // do-nothing baseline: every pair conflicts
    for (int i = 0; i < M; i++) {
        eu[i] = inf.readInt();
        ev[i] = inf.readInt();
        ew[i] = inf.readInt();
        B += ew[i];
    }

    // read participant assignment: exactly N integers in [1,K]
    vector<int> col(N + 1);
    for (int j = 1; j <= N; j++)
        col[j] = ouf.readInt(1, K, "c_j");
    if (!ouf.seekEof())
        quitf(_wa, "trailing tokens after the N program assignments");

    long long F = 0;
    for (int i = 0; i < M; i++)
        if (col[eu[i]] == col[ev[i]])
            F += ew[i];

    if (B <= 0)
        quitf(_fail, "internal: non-positive baseline B=%lld", B);

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
