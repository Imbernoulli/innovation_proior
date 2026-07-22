#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Couple oscillators just enough to synchronize".
//
// Input:  N M R C T W ; then N reals x_1..x_N (initial states); then M lines
//   "u v cap" (candidate coupling edge, 1-indexed, undirected, capacity cap).
//
// Output: M reals c_1..c_M, c_e = coupling strength assigned to edge e (same
//   order as the input edge list).
//
// Feasibility (MUST hold, else score 0):
//   - every c_e finite and in [0, cap_e]
//   - for every node i, sum of c_e over edges incident to i is <= 1.0
//     (the coupling-DEGREE budget: this keeps the coupled-map update a convex
//      combination, so states stay in the map's range -- a physical bound,
//      not a per-test input)
//   - sum of all c_e <= C  (the total coupling budget)
//
// Dynamics (coupled logistic maps, diffusive coupling): let deg_i = sum of
// c_e over i's incident edges. Synchronously, for t = 0..T-1:
//   f_i(t) = R * x_i(t) * (1 - x_i(t))
//   x_i(t+1) = (1 - deg_i) * f_i(t) + sum_{e=(i,j)} c_e * f_j(t)
// This is a convex combination of f-values (weights are all >= 0 and sum to
// 1 because deg_i <= 1), so x_i(t+1) always stays in [0, max f] subset [0,1].
//
// Objective (MAX): over the trailing W steps (t = T-W .. T-1), let xbar(t) be
// the mean of x_1(t)..x_N(t), and let dev_i = RMS_t( x_i(t) - xbar(t) ). Each
// node contributes
//   score_i = BASE + (1-BASE) * exp(-dev_i / TAU)      (BASE=0.05, TAU=0.12)
// F = sum_i score_i, maximized when every node tracks the network mean
// tightly and throughout the trailing window (not just momentarily).
//
// Baseline B (checker-computed): F for the all-zero coupling assignment
// (every oscillator evolves independently). B > 0 always since score_i >=
// BASE*N > 0. This is exactly what the trivial reference reproduces.
// Score (max): sc = min(1000, 100*F/max(1e-9,B)); ratio = sc/1000.
// -----------------------------------------------------------------------------

static const double DEGCAP = 1.0;
// Two tolerances, deliberately different:
//  - EDGE_EPS guards a SINGLE value against a SINGLE bound (one edge's c_e
//    vs. its own cap, or vs. 0). There is no summation here, so this must
//    stay near machine precision -- anything looser would let a submission
//    buy real extra score by quietly exceeding a cap (the coupling-degree
//    bound is exactly what keeps the dynamics physically valid, so even a
//    tiny systematic excess on many edges is exploitable).
//  - AGG_EPS guards a SUM of up to a few thousand edge weights (a node's
//    total degree, or the total budget spent) against a bound, where
//    genuine floating-point accumulation from iterative water-filling
//    (M <= 3600 terms) needs headroom well above machine epsilon -- but
//    still far below any real violation (the invalid reference overshoots
//    a node's degree by ~0.45, four orders of magnitude more than this).
static const double EDGE_EPS = 1e-9;
static const double AGG_EPS = 1e-4;
static const double BASE = 0.05;
static const double TAU = 0.12;

int N, M, T, W;
double R, C;
vector<double> X0;
vector<int> EU, EV;
vector<double> ECap;

double simulateF(const vector<double> &c) {
    vector<double> deg(N + 1, 0.0);
    vector<vector<pair<int,double>>> adj(N + 1); // (other endpoint, c_e)
    for (int e = 0; e < M; e++) {
        int u = EU[e], v = EV[e];
        deg[u] += c[e]; deg[v] += c[e];
        adj[u].push_back({v, c[e]});
        adj[v].push_back({u, c[e]});
    }
    vector<double> x = X0; // x[1..N]
    vector<vector<double>> hist; // trailing W snapshots
    hist.reserve(W);
    vector<double> f(N + 1), nx(N + 1);
    for (int t = 1; t <= T; t++) {
        for (int i = 1; i <= N; i++) f[i] = R * x[i] * (1.0 - x[i]);
        for (int i = 1; i <= N; i++) nx[i] = (1.0 - deg[i]) * f[i];
        for (int e = 0; e < M; e++) {
            double ce = c[e];
            if (ce > 0.0) {
                int u = EU[e], v = EV[e];
                nx[u] += ce * f[v];
                nx[v] += ce * f[u];
            }
        }
        x.swap(nx);
        if (t > T - W) hist.push_back(x); // stores x[0..N], x[0] unused
    }
    int Wn = (int)hist.size();
    vector<double> xbar(Wn, 0.0);
    for (int k = 0; k < Wn; k++) {
        double s = 0.0;
        for (int i = 1; i <= N; i++) s += hist[k][i];
        xbar[k] = s / N;
    }
    double F = 0.0;
    for (int i = 1; i <= N; i++) {
        double s2 = 0.0;
        for (int k = 0; k < Wn; k++) {
            double d = hist[k][i] - xbar[k];
            s2 += d * d;
        }
        double dev = sqrt(s2 / max(1, Wn));
        double score_i = BASE + (1.0 - BASE) * exp(-dev / TAU);
        F += score_i;
    }
    return F;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    N = inf.readInt();
    M = inf.readInt();
    R = inf.readDouble();
    C = inf.readDouble();
    T = inf.readInt();
    W = inf.readInt();

    X0.assign(N + 1, 0.0);
    for (int i = 1; i <= N; i++) X0[i] = inf.readDouble();

    EU.resize(M); EV.resize(M); ECap.resize(M);
    for (int e = 0; e < M; e++) {
        EU[e] = inf.readInt();
        EV[e] = inf.readInt();
        ECap[e] = inf.readDouble();
    }

    // ---- internal baseline B: zero coupling ----
    vector<double> zero(M, 0.0);
    double B = simulateF(zero);
    if (B <= 0.0) B = 1e-9; // guaranteed positive (>= N*BASE) but guard anyway

    // ---- replay participant's coupling assignment ----
    vector<double> c(M);
    for (int e = 0; e < M; e++) {
        double v = ouf.readDouble();
        if (!isfinite(v)) quitf(_wa, "edge %d: coupling value is not finite", e + 1);
        if (v < -EDGE_EPS || v > ECap[e] + EDGE_EPS)
            quitf(_wa, "edge %d: coupling %.9f out of range [0, %.6f]", e + 1, v, ECap[e]);
        c[e] = min(ECap[e], max(0.0, v)); // clamp the sub-EDGE_EPS slack so simulation never sees an over-cap value
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after the %d coupling values", M);

    vector<double> deg(N + 1, 0.0);
    double total = 0.0;
    for (int e = 0; e < M; e++) {
        deg[EU[e]] += c[e];
        deg[EV[e]] += c[e];
        total += c[e];
    }
    for (int i = 1; i <= N; i++)
        if (deg[i] > DEGCAP + AGG_EPS)
            quitf(_wa, "node %d coupling-degree %.9f exceeds cap %.6f", i, deg[i], DEGCAP);
    if (total > C + AGG_EPS)
        quitf(_wa, "total coupling %.9f exceeds budget C=%.6f", total, C);

    double F = simulateF(c);

    double sc = min(1000.0, 100.0 * F / max(1e-9, B));
    quitp(sc / 1000.0, "OK F=%.6f B=%.6f Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
