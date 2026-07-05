#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int P = inf.readInt();
    vector<long long> C(P);
    for (int j = 0; j < P; j++) C[j] = inf.readLong();
    vector<vector<long long>> v(N, vector<long long>(P)), w(N, vector<long long>(P));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < P; j++) {
            v[i][j] = inf.readLong();
            w[i][j] = inf.readLong();
        }

    // ---- read participant assignment, validate feasibility ----
    vector<long long> used(P, 0);
    long long F = 0;
    for (int i = 0; i < N; i++) {
        int a = ouf.readInt(0, P, format("a[%d]", i + 1));
        if (a >= 1) {
            int j = a - 1;
            used[j] += w[i][j];
            if (used[j] > C[j])
                quitf(_wa, "pool %d overfilled: used %lld > capacity %lld", a, used[j], C[j]);
            F += v[i][j];
        }
    }
    if (!ouf.seekEof())
        quitf(_wa, "trailing tokens after %d assignments", N);

    // ---- internal baseline B: first-fit in input / pool-index order ----
    vector<long long> rem(P);
    for (int j = 0; j < P; j++) rem[j] = C[j];
    long long B = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < P; j++) {
            if (w[i][j] <= rem[j]) {
                rem[j] -= w[i][j];
                B += v[i][j];
                break;
            }
        }
    }
    if (B < 1) B = 1;

    double sc = min(1000.0, 100.0 * (double)F / (double)max(1LL, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
