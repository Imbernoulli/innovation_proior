#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);
    if (t < 1) t = 1;
    if (t > 10) t = 10;

    // Difficulty ladder: sizes grow, capacities tighten, structure skews.
    int Ns[11] = {0, 5, 12, 25, 40, 60, 90, 130, 180, 240, 300};
    int Ps[11] = {0, 2,  3,  4,  5,  6,  8, 12,  16,  20,  25};
    int N = Ns[t], P = Ps[t];

    const int K = 5; // number of strain/pool types

    // pool types
    vector<int> pt(P);
    for (int j = 0; j < P; j++) pt[j] = rnd.next(0, K - 1);

    // salmon: type, base value, base weight
    vector<int> st(N), g(N);
    vector<vector<int>> w(N, vector<int>(P)), v(N, vector<int>(P));
    long long sumW = 0;
    for (int i = 0; i < N; i++) {
        st[i] = rnd.next(0, K - 1);
        g[i]  = rnd.next(20, 100);
        int bw = rnd.next(8, 45);
        for (int j = 0; j < P; j++) {
            // weight varies a little per pool (flow / temperature)
            int ww = bw + rnd.next(-4, 10);
            if (ww < 5) ww = 5;
            if (ww > 55) ww = 55;
            w[i][j] = ww;
            sumW += ww;
        }
    }
    double avgW = (double)sumW / max(1, N * P);

    // value depends on strain/pool match: exact strain x5, adjacent x2, else x1
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < P; j++) {
            int diff = abs(pt[j] - st[i]);
            int bonus = (diff == 0) ? 5 : (diff == 1 ? 2 : 1);
            long long val = (long long)g[i] * bonus;
            if (val > 1000) val = 1000;
            v[i][j] = (int)val;
        }
    }

    // capacity tightness decreases with t (harder), never below a floor that
    // guarantees the first school can be placed (so baseline B > 0).
    double alpha = 0.65 - 0.042 * (t - 1); // ~0.65 down to ~0.27
    if (alpha < 0.22) alpha = 0.22;
    long long totalCap = (long long)llround(alpha * N * avgW);
    long long floorCap = (long long)P * 60; // > max weight (55)
    if (totalCap < floorCap) totalCap = floorCap;

    vector<long long> C(P);
    long long base = totalCap / P;
    int skew = 5 + 4 * t; // uneven pools grow with t
    long long assigned = 0;
    for (int j = 0; j < P; j++) {
        long long cj = base + rnd.next(-skew, skew);
        if (cj < 60) cj = 60;
        if (cj > 200000) cj = 200000;
        C[j] = cj;
        assigned += cj;
    }

    // print
    printf("%d %d\n", N, P);
    for (int j = 0; j < P; j++) printf("%lld%c", C[j], j + 1 == P ? '\n' : ' ');
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < P; j++) {
            printf("%d %d", v[i][j], w[i][j]);
            printf("%c", j + 1 == P ? '\n' : ' ');
        }
    }
    return 0;
}
