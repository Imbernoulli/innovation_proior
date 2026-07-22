#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Living-Hinge Curvature Fit"  (ligament-compliance-hinge).
//
// Input:  m k S M C
//         m integers target_1 .. target_m   (target curvature * 1000, rounded)
//
// Output (participant): m integers r_1 .. r_m, S <= r_i <= k, the remaining net
//   section (ligament count) of each column after slitting.
//
// Feasibility: every r_i in [S,k]; total material removed sum(k-r_i) <= C.
//
// Objective (MIN):  F = sum_i | M / r_i^3 - target_i/1000 |.
// Baseline B (checker-computed do-nothing): r_i = k for every column (0 cuts,
//   always feasible since C >= 0 and k is in [S,k]).  This is exactly what the
//   trivial reference reproduces -> ratio 0.1.
// Score (min): sc = min(1000, 100 * B / max(1, F)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int m = inf.readInt();
    int k = inf.readInt();
    int S = inf.readInt();
    ll M = inf.readLong();
    ll C = inf.readLong();
    vector<ll> target(m);
    for (int i = 0; i < m; i++) target[i] = inf.readLong();

    // ---- internal baseline B: do-nothing (r_i = k for all columns) ----
    double curvK = (double)M / ((double)k * (double)k * (double)k);
    double B_d = 0.0;
    for (int i = 0; i < m; i++) {
        double t = (double)target[i] / 1000.0;
        B_d += fabs(curvK - t);
    }

    // ---- read participant output (strict feasibility) ----
    vector<int> r(m);
    ll totalCost = 0;
    for (int i = 0; i < m; i++) {
        int v = ouf.readInt(S, k, format("r_%d", i + 1).c_str());
        r[i] = v;
        totalCost += (ll)(k - v);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after r_%d", m);
    if (totalCost > C)
        quitf(_wa, "slit budget exceeded: used %lld > C=%lld", totalCost, C);

    double F_d = 0.0;
    for (int i = 0; i < m; i++) {
        double curv = (double)M / ((double)r[i] * (double)r[i] * (double)r[i]);
        double t = (double)target[i] / 1000.0;
        F_d += fabs(curv - t);
    }

    const double SCALE = 1000000.0;
    ll Fi = (ll)llround(F_d * SCALE);
    ll Bi = (ll)llround(B_d * SCALE);
    if (Bi < 1) Bi = 1;

    double sc = min(1000.0, 100.0 * (double)Bi / (double)max(1LL, Fi));
    quitp(sc / 1000.0, "OK F=%lld B=%lld cost=%lld/%lld Ratio: %.6f",
          Fi, Bi, totalCost, C, sc / 1000.0);
    return 0;
}
