#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int P = inf.readInt();
    vector<long long> C(P);
    vector<long long> K(P);
    for (int j = 0; j < P; j++) {
        C[j] = inf.readLong();
        K[j] = inf.readLong();
    }
    vector<vector<long long>> val(N, vector<long long>(P)), vol(N, vector<long long>(P));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < P; j++) {
            val[i][j] = inf.readLong();
            vol[i][j] = inf.readLong();
        }

    // ---- read participant assignment, validate feasibility ----
    vector<long long> usedVol(P, 0);
    vector<long long> usedCnt(P, 0);
    long long F = 0;
    for (int i = 0; i < N; i++) {
        int a = ouf.readInt(0, P, format("a[%d]", i + 1));
        if (a >= 1) {
            int j = a - 1;
            usedVol[j] += vol[i][j];
            usedCnt[j] += 1;
            if (usedVol[j] > C[j])
                quitf(_wa, "gate %d discharge exceeded: used %lld > capacity %lld", a, usedVol[j], C[j]);
            if (usedCnt[j] > K[j])
                quitf(_wa, "gate %d valve limit exceeded: %lld zones > K %lld", a, usedCnt[j], K[j]);
            F += val[i][j];
        }
    }
    if (!ouf.seekEof())
        quitf(_wa, "trailing tokens after %d assignments", N);

    // ---- internal baseline B: first-fit in input / gate-index order ----
    vector<long long> remVol(P), remCnt(P);
    for (int j = 0; j < P; j++) { remVol[j] = C[j]; remCnt[j] = K[j]; }
    long long B = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < P; j++) {
            if (remCnt[j] >= 1 && vol[i][j] <= remVol[j]) {
                remVol[j] -= vol[i][j];
                remCnt[j] -= 1;
                B += val[i][j];
                break;
            }
        }
    }
    if (B < 1) B = 1;

    double sc = min(1000.0, 100.0 * (double)F / (double)max(1LL, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
