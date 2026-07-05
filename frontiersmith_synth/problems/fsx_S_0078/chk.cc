#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// chk.cc -- scorer for Data-Center Cooling Assignment (capacitated GAP, maximization).
// Validates: exactly N integers a_i in [0,M]; per-unit total assigned cooling load <= C_j.
// Objective F = sum of v_i over cooled racks (a_i != 0).
// Baseline B = value of first-fit-in-index-order: rack i=1..N assigned to the first unit
//   j=1..M whose remaining capacity >= d[i][j] (else uncooled).  Always positive because
//   every d[i][j] <= C_j.  Score  ratio = min(1, (F / max(1,B)) / 10).

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int M = inf.readInt();
    vector<long long> v(N);
    for (int i = 0; i < N; i++) v[i] = inf.readInt();
    vector<long long> C(M);
    for (int j = 0; j < M; j++) C[j] = inf.readLong();
    vector<vector<long long>> d(N, vector<long long>(M));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++) d[i][j] = inf.readLong();

    // ---- read participant assignment ----
    vector<long long> load(M, 0);
    long long F = 0;
    for (int i = 0; i < N; i++) {
        int a = ouf.readInt(0, M, format("a[%d]", i + 1).c_str());
        if (a != 0) {
            int j = a - 1;
            load[j] += d[i][j];
            if (load[j] > C[j])
                quitf(_wa, "unit %d overloaded: load %lld exceeds capacity %lld",
                      j + 1, load[j], C[j]);
            F += v[i];
        }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after the %d assignments", N);

    // ---- baseline B: first-fit in index order ----
    vector<long long> rem(C);
    long long B = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            if (rem[j] >= d[i][j]) { rem[j] -= d[i][j]; B += v[i]; break; }
        }
    }
    if (B <= 0) B = 1;

    double sc = min(1000.0, 100.0 * (double)F / (double)max(1LL, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
