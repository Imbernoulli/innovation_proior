#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int T = inf.readInt();
    int K = inf.readInt();
    long long D = inf.readInt();

    vector<long long> spot(T);
    for (int j = 0; j < T; j++) spot[j] = inf.readInt();

    vector<long long> R(N), lam(N);
    for (int i = 0; i < N; i++) R[i] = inf.readInt();
    for (int i = 0; i < N; i++) lam[i] = inf.readInt();

    vector<vector<char>> avail(N, vector<char>(T));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < T; j++)
            avail[i][j] = (char)inf.readInt();

    // baseline B = do-nothing (all idle) cost
    long long B = 0;
    for (int i = 0; i < N; i++) B += lam[i] * R[i];

    // read participant assignment: N rows of T modes in {0,1,2}
    vector<int> rangerCount(T, 0);
    long long cost = 0, shortfall = 0;
    for (int i = 0; i < N; i++) {
        long long work = 0;
        for (int j = 0; j < T; j++) {
            int m = ouf.readInt(0, 2, "mode");
            if (m == 1) {
                if (!avail[i][j])
                    quitf(_wa, "tower %d step %d: spot chosen but volunteer not available", i + 1, j + 1);
                cost += spot[j];
                work++;
            } else if (m == 2) {
                rangerCount[j]++;
                if (rangerCount[j] > K)
                    quitf(_wa, "step %d: more than K=%d towers in ranger mode", j + 1, K);
                cost += D;
                work++;
            }
        }
        long long miss = R[i] - work;
        if (miss > 0) shortfall += lam[i] * miss;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output");

    long long F = cost + shortfall;
    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
