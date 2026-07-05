#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);
    if (t < 1) t = 1;
    if (t > 10) t = 10;

    // Difficulty ladder: sizes grow, capacities & valve limits tighten, structure skews.
    int Ns[11] = {0, 5, 12, 40, 120, 300, 600, 1000, 1400, 1700, 2000};
    int Ps[11] = {0, 2,  3,  6,  10,  16,  24,   34,   44,   54,   60};
    int N = Ns[t], P = Ps[t];

    const int K = 6; // number of hydro/quality types

    // gate types
    vector<int> gtype(P);
    for (int j = 0; j < P; j++) gtype[j] = rnd.next(0, K - 1);

    // zones: type + base value magnitude
    vector<int> ztype(N), g(N), bw(N);
    for (int i = 0; i < N; i++) {
        ztype[i] = rnd.next(0, K - 1);
        g[i]     = rnd.next(20, 100);
        bw[i]    = rnd.next(10, 60); // base demand
    }

    // volume: base demand + conveyance loss growing with type distance
    vector<vector<int>> vol(N, vector<int>(P)), val(N, vector<int>(P));
    long long sumVol = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < P; j++) {
            int diff = abs(gtype[j] - ztype[i]);
            int loss = diff * rnd.next(3, 10) + rnd.next(0, 6);
            int vv = bw[i] + loss;
            if (vv < 1) vv = 1;
            if (vv > 1000) vv = 1000;
            vol[i][j] = vv;
            sumVol += vv;
            // value: better when types match (pressure / quality suited)
            int bonus = (diff == 0) ? 5 : (diff == 1 ? 3 : 1);
            long long vl = (long long)g[i] * bonus;
            if (vl > 1000) vl = 1000;
            val[i][j] = (int)vl;
        }
    }
    double avgVol = (double)sumVol / max(1, N * P);

    // discharge capacity: tightness decreases (harder) with t
    double alpha = 0.60 - 0.040 * (t - 1); // ~0.60 down to ~0.24
    if (alpha < 0.20) alpha = 0.20;
    long long totalCap = (long long)llround(alpha * N * avgVol);
    long long floorCapPer = 1000; // >= max possible vol, guarantees first zone fits -> B > 0
    if (totalCap < floorCapPer * P) totalCap = floorCapPer * P;

    vector<long long> C(P);
    long long baseC = totalCap / P;
    for (int j = 0; j < P; j++) {
        // skewed capacities (some big reservoirs, some small)
        double f = 0.4 + rnd.next(0, 1200) / 1000.0; // 0.4 .. 1.6
        long long cj = (long long)llround(baseC * f);
        if (cj < floorCapPer) cj = floorCapPer;
        if (cj > 1000000) cj = 1000000;
        C[j] = cj;
    }

    // valve limits: total slots ~ beta*N, tightening with t; skewed per gate; >=1
    double beta = 1.7 - 0.07 * (t - 1); // ~1.70 down to ~1.07
    if (beta < 1.0) beta = 1.0;
    long long totalK = (long long)llround(beta * N);
    if (totalK < P) totalK = P; // at least 1 per gate on average
    vector<long long> Kv(P);
    long long baseK = max<long long>(1, totalK / P);
    for (int j = 0; j < P; j++) {
        double f = 0.4 + rnd.next(0, 1200) / 1000.0;
        long long kj = (long long)llround(baseK * f);
        if (kj < 1) kj = 1;
        if (kj > N) kj = N;
        Kv[j] = kj;
    }
    // ensure gate 1 can accept at least one zone (belt-and-suspenders for B>0)
    if (Kv[0] < 1) Kv[0] = 1;
    if (C[0] < floorCapPer) C[0] = floorCapPer;

    // print
    printf("%d %d\n", N, P);
    for (int j = 0; j < P; j++) printf("%lld %lld%c", C[j], Kv[j], j + 1 == P ? '\n' : ' ');
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < P; j++) {
            printf("%d %d", val[i][j], vol[i][j]);
            printf("%c", j + 1 == P ? '\n' : ' ');
        }
    }
    return 0;
}
