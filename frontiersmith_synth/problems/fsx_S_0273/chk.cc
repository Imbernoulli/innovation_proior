#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// chk.cc -- scorer for satellite ground-station channel separation (weighted T-coloring / FAP).
// Minimization. Baseline B = interference of the all-same-channel assignment = sum_i w_i*g_i
// (every pair is co-channel, cost w*g). Participant cost
//   F = sum_i w_i * max(0, g_i - |c_u - c_v|).
// Since |c_u-c_v| >= 0, F <= B always, so a feasible output scores ratio >= 0.1.
// ratio = min(1, (B / max(1,F)) / 10).
int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int M = inf.readInt();
    int K = inf.readInt();

    vector<int> eu(M), ev(M), eg(M);
    vector<long long> ew(M);
    long long B = 0;                       // do-nothing baseline: every pair co-channel
    for (int i = 0; i < M; i++) {
        eu[i] = inf.readInt();
        ev[i] = inf.readInt();
        eg[i] = inf.readInt();
        ew[i] = inf.readInt();
        B += ew[i] * (long long)eg[i];
    }

    // read participant assignment: exactly N integers in [1,K]
    vector<int> col(N + 1);
    for (int j = 1; j <= N; j++)
        col[j] = ouf.readInt(1, K, "c_j");
    if (!ouf.seekEof())
        quitf(_wa, "trailing tokens after the N channel assignments");

    long long F = 0;
    for (int i = 0; i < M; i++) {
        int d = abs(col[eu[i]] - col[ev[i]]);
        int deficit = eg[i] - d;
        if (deficit > 0)
            F += ew[i] * (long long)deficit;
    }

    if (B <= 0)
        quitf(_fail, "internal: non-positive baseline B=%lld", B);

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
