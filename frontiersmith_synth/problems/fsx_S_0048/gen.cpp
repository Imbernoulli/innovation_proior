#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Offline data-center cooling schedule ("Coolant Rush").
// testId is a difficulty/structure ladder:
//   testId 1  -> tiny (example scale): 2 zones, 12 steps
//   testId 10 -> many zones, long horizon, strong price valleys/peaks (adversarial)

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int tid = atoi(argv[1]);

    int Z, T;
    if (tid == 1) { Z = 2; T = 12; }
    else { Z = min(8, 2 + tid / 2); T = 40 * tid; }   // up to Z=7, T=400

    // ---- zone limits and starting temperatures ----
    vector<int> Cap(Z), init(Z);
    for (int j = 0; j < Z; j++) {
        Cap[j] = rnd.next(50, 90);
        init[j] = max(1, Cap[j] / 3);                 // room to pre-cool below and to coast above
    }

    // ---- heat traces: diurnal workload bump + noise, clamped to [0,6] ----
    const int period = 24;
    vector<vector<int>> h(Z, vector<int>(T, 0));
    for (int j = 0; j < Z; j++) {
        int amp = rnd.next(2, 4);
        int base = rnd.next(0, 2);
        for (int t = 0; t < T; t++) {
            int ph = t % period;
            int day = (ph >= 8 && ph < 20) ? amp : 0;
            int val = base + day + rnd.next(0, 2);
            if (val > 6) val = 6;
            if (val < 0) val = 0;
            h[j][t] = val;
        }
    }
    h[0][0] = max(h[0][0], 1);                         // guarantee total heat > 0

    // ---- spot electricity prices: cheap valleys with expensive diurnal peaks ----
    vector<int> p(T);
    int pkStart = 16 + (tid % 3);                      // 16..18
    int pkLen = 4 + (tid % 3);                         // 4..6
    for (int t = 0; t < T; t++) {
        int ph = t % period;
        bool peak = (ph >= pkStart && ph < pkStart + pkLen);
        if (peak) p[t] = rnd.next(300, 1000);
        else {
            p[t] = rnd.next(1, 30);
            if (rnd.next(0, 99) < 3) p[t] = rnd.next(200, 600);  // rare off-peak spike
        }
    }

    // ---- fixed startup/run charge ----
    int s = rnd.next(40, 200);

    // ---- shared plant capacity: reactive is feasible, with headroom to pre-cool ----
    int maxload = 0;
    for (int t = 0; t < T; t++) {
        int X = 0;
        for (int j = 0; j < Z; j++) X += h[j][t];
        maxload = max(maxload, X);
    }
    if (maxload < 1) maxload = 1;
    int C = maxload + max(Z * 3, maxload);             // ~2x max instantaneous load

    // ---- emit ----
    printf("%d %d %d %d\n", Z, T, C, s);
    for (int t = 0; t < T; t++) printf("%d%c", p[t], t == T - 1 ? '\n' : ' ');
    for (int j = 0; j < Z; j++) printf("%d%c", Cap[j], j == Z - 1 ? '\n' : ' ');
    for (int j = 0; j < Z; j++) printf("%d%c", init[j], j == Z - 1 ? '\n' : ' ');
    for (int j = 0; j < Z; j++)
        for (int t = 0; t < T; t++)
            printf("%d%c", h[j][t], t == T - 1 ? '\n' : ' ');
    return 0;
}
