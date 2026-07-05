#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// gen.cpp -- Data-Center Cooling Assignment (capacitated generalized assignment problem).
// testId is a difficulty/structure ladder: tiny at 1 growing to N=1000, M=25 at 10.
// Structure varies: cooling load d[i][j] = h_i * airflow_factor(dist(rack_i, unit_j)) so a
// rack is cheap on nearby units and expensive on far ones (real per-unit assignment matters);
// even tests place units in clusters (some racks stranded far from all units); the capacity
// tightness rho (fraction of average load that fits in total) cycles tight/medium/loose so
// packing is genuinely binding.  Values v_i are drawn independently of heat so density-based
// heuristics differ from value-based ones.  Every d[i][j] <= C_j is guaranteed.
int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    int idx = min(max(testId, 1), 10);

    int Ns[10] = {3, 20, 45, 90, 160, 260, 400, 600, 800, 1000};
    int Ms[10] = {2, 4,  5,  6,  8,   10,  12,  16,  20,  25};
    int N = Ns[idx - 1];
    int M = Ms[idx - 1];
    const int G = 1000;             // hall grid size
    const int Vmax = 1000;          // max rack value
    bool clustered = (idx % 2 == 0);
    double rho = (idx % 3 == 0) ? 0.35 : (idx % 3 == 1 ? 0.55 : 0.75);

    // ---- rack positions + base heat ----
    vector<int> rx(N), ry(N), h(N), v(N);
    for (int i = 0; i < N; i++) {
        rx[i] = rnd.next(0, G);
        ry[i] = rnd.next(0, G);
        h[i]  = rnd.next(10, 100);
        v[i]  = rnd.next(1, Vmax);
    }

    // ---- unit positions ----
    vector<int> ux(M), uy(M);
    if (clustered) {
        int nC = max(1, M / 3);
        vector<pair<int,int>> ctr;
        for (int c = 0; c < nC; c++) ctr.push_back({rnd.next(0, G), rnd.next(0, G)});
        int spread = max(30, G / (4 * nC));
        for (int j = 0; j < M; j++) {
            auto& c = ctr[rnd.next(0, nC - 1)];
            ux[j] = min(G, max(0, c.first  + rnd.next(-spread, spread)));
            uy[j] = min(G, max(0, c.second + rnd.next(-spread, spread)));
        }
    } else {
        for (int j = 0; j < M; j++) { ux[j] = rnd.next(0, G); uy[j] = rnd.next(0, G); }
    }

    // ---- cooling-load matrix d[i][j] = h_i * (1 + 4*dist/(2G)) ----
    // dist is Manhattan (0..2G); factor ranges 1..3, so a rack costs 1x-3x by distance.
    vector<vector<long long>> d(N, vector<long long>(M));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            long long dist = llabs((long long)rx[i] - ux[j]) + llabs((long long)ry[i] - uy[j]);
            double factor = 1.0 + 2.0 * (double)dist / (double)(2 * G);
            long long val = (long long)llround((double)h[i] * factor);
            if (val < 1) val = 1;
            d[i][j] = val;
        }
    }

    // ---- capacities: C_j >= max_i d[i][j] (single rack always fits), sized so total
    //      capacity is roughly rho * (load if every rack took its average unit). ----
    double avgLoad = 0.0;
    for (int i = 0; i < N; i++) {
        long long s = 0; for (int j = 0; j < M; j++) s += d[i][j];
        avgLoad += (double)s / (double)M;
    }
    long long baseCap = (long long)llround(rho * avgLoad / (double)M);
    if (baseCap < 1) baseCap = 1;
    vector<long long> C(M, baseCap);
    for (int j = 0; j < M; j++) {
        long long mx = 1;
        for (int i = 0; i < N; i++) mx = max(mx, d[i][j]);
        C[j] = max(C[j], mx);
    }

    // ---- emit ----
    printf("%d %d\n", N, M);
    for (int i = 0; i < N; i++) printf("%d%c", v[i], i + 1 < N ? ' ' : '\n');
    for (int j = 0; j < M; j++) printf("%lld%c", C[j], j + 1 < M ? ' ' : '\n');
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++) printf("%lld%c", d[i][j], j + 1 < M ? ' ' : '\n');
    return 0;
}
