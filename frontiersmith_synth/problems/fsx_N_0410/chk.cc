#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// chk.cc -- scorer for Orbital Debris Cleanup: Shared-Aperture Sweeper Deployment.
// Maximization. Reads the fleet (aperture index t + deployed cluster indices), validates:
//   * t in [1,K];  m in [0,N];  indices distinct in [1,N];
//   * NO collisions at r=R[t]: every deployed pair has squared centre distance >= 4*r*r
//     (verified with a spatial grid of cell 2r, so O(m) expected -- fast on 20 MB inputs).
// Objective F = sum_i v_i(r) + D*(#distinct bands),  v_i(r)=w_i*r - c_i*r^2 - a_i.
// Baseline B = max(1, D + max over all i,t of v_i(R[t]))  (the best single sweeper).
// ratio = clamp(100*F/B, 0, 1000)/1000.  Single-best fleet -> ratio 0.1.

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int K = inf.readInt();
    long long D = inf.readLong();
    int NB = inf.readInt();
    vector<long long> R(K);
    for (int t = 0; t < K; t++) R[t] = inf.readLong();

    vector<long long> X(N), Y(N), W(N), Cc(N), Aa(N);
    vector<int> Bd(N);
    for (int i = 0; i < N; i++) {
        X[i] = inf.readLong();
        Y[i] = inf.readLong();
        W[i] = inf.readLong();
        Cc[i] = inf.readLong();
        Aa[i] = inf.readLong();
        Bd[i] = inf.readInt();
    }

    // ---- baseline B: best single sweeper over all clusters and apertures ----
    long long bestSingle = LLONG_MIN;
    for (int i = 0; i < N; i++)
        for (int t = 0; t < K; t++) {
            long long r = R[t];
            long long v = W[i] * r - Cc[i] * r * r - Aa[i];
            if (v > bestSingle) bestSingle = v;
        }
    long long B = D + bestSingle;
    if (B < 1) B = 1;

    // ---- read participant fleet ----
    int t1 = ouf.readInt(1, K, "t");
    long long r = R[t1 - 1];
    int m = ouf.readInt(0, N, "m");

    vector<int> sel(m);
    vector<char> usedIdx(N + 1, 0);
    for (int k = 0; k < m; k++) {
        int i = ouf.readInt(1, N, "cluster_index");
        if (usedIdx[i]) quitf(_wa, "cluster %d deployed more than once", i);
        usedIdx[i] = 1;
        sel[k] = i - 1;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after the fleet listing");

    // ---- collision check via spatial grid (cell = 2r) ----
    long long cell = 2 * r;                    // r >= 1 so cell >= 2
    long long thr = 4 * r * r;                 // overlap iff dist^2 < thr
    unordered_map<long long, vector<int>> grid;
    grid.reserve(m * 2 + 16);
    auto keyOf = [&](long long cx, long long cy) -> long long {
        return cx * 4000003LL + cy;            // altitude bands are small; coord/cell < ~30000
    };
    for (int k = 0; k < m; k++) {
        int i = sel[k];
        long long cx = X[i] / cell, cy = Y[i] / cell;
        for (long long dx = -1; dx <= 1; dx++)
            for (long long dy = -1; dy <= 1; dy++) {
                auto it = grid.find(keyOf(cx + dx, cy + dy));
                if (it == grid.end()) continue;
                for (int j : it->second) {
                    long long ddx = X[i] - X[j], ddy = Y[i] - Y[j];
                    long long d2 = ddx * ddx + ddy * ddy;
                    if (d2 < thr)
                        quitf(_wa, "clusters %d and %d collide at aperture R[%d]=%lld (dist^2=%lld < %lld)",
                              i + 1, j + 1, t1, r, d2, thr);
                }
            }
        grid[keyOf(cx, cy)].push_back(i);
    }

    // ---- objective F ----
    long long sumv = 0;
    set<int> bands;
    for (int k = 0; k < m; k++) {
        int i = sel[k];
        sumv += W[i] * r - Cc[i] * r * r - Aa[i];
        bands.insert(Bd[i]);
    }
    long long F = sumv + D * (long long)bands.size();

    // ---- score ----
    double raw = 100.0 * (double)F / (double)B;
    if (!isfinite(raw)) raw = 0.0;
    double sc = raw;
    if (sc < 0.0) sc = 0.0;
    if (sc > 1000.0) sc = 1000.0;
    quitp(sc / 1000.0, "OK F=%lld B=%lld bands=%d m=%d Ratio: %.6f",
          F, B, (int)bands.size(), m, sc / 1000.0);
    return 0;
}
