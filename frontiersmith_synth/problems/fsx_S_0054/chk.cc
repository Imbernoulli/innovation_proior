#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int M = inf.readInt();
    vector<long long> C(N);
    for (int j = 0; j < N; j++) C[j] = inf.readLong();
    // t[i][j], v[i][j] stored flat
    vector<long long> t((size_t)M * N), v((size_t)M * N);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            t[(size_t)i * N + j] = inf.readLong();
            v[(size_t)i * N + j] = inf.readLong();
        }

    // read participant assignment, validate strictly
    vector<long long> used(N, 0);
    long long F = 0;
    for (int i = 0; i < M; i++) {
        int a = ouf.readInt(0, N, "assign"); // 0 = uncovered, else 1..N
        if (a > 0) {
            int j = a - 1;
            used[j] += t[(size_t)i * N + j];
            if (used[j] > C[j])
                quitf(_wa, "tracer %d over budget: used=%lld cap=%lld (at cluster %d)",
                      a, used[j], C[j], i + 1);
            F += v[(size_t)i * N + j];
        }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output");

    // internal baseline B: first-fit in cluster order, lowest-index affordable tracer
    vector<long long> rem = C;
    long long B = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            long long cost = t[(size_t)i * N + j];
            if (rem[j] >= cost) {
                rem[j] -= cost;
                B += v[(size_t)i * N + j];
                break;
            }
        }
    }
    if (B < 1) B = 1;

    double sc = min(1000.0, 100.0 * (double)F / (double)max(1LL, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
