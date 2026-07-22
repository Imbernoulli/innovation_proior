#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// -----------------------------------------------------------------------------
// Checker / scorer for "Express Tunnels to the Depot"   family: hub-dilution-shortcuts
//
// Input:  n m0 t k M ; then m0 lines "u v" (original tree edges), then M lines
//         "a b" (candidate shortcut endpoints, 1-indexed; index = 1-based line#).
// Output: c  (0<=c<=k)   then c DISTINCT candidate indices in [1,M].
//
// Objective (MIN): after adding the chosen candidate edges (mass m = m0+c), let
//   R(u,t) be the u<->t effective resistance of the resulting (multi)graph under
//   unit edge conductance. The worst-case round-trip commute time to the depot is
//     C(u,t) = 2 * m * R(u,t)          (exact commute-time <-> resistance identity
//                                        for the simple random walk on a graph
//                                        with m edges: H(u,t)+H(t,u) = 2 m R(u,t)).
//   Objective  F = max_{u != t} C(u,t).  Minimize F.
//
// R(u,t) for EVERY u is obtained in one shot: ground the Laplacian at t (drop its
// row/column) to get L_t (size (n-1)x(n-1), positive definite since the graph is
// connected); then R(u,t) = (L_t^{-1})_{uu}, i.e. the diagonal of the inverse.
//
// Internal baseline B (checker-computed, MIN-convention "do nothing"): F on the
// ORIGINAL graph with zero shortcuts added (m=m0). Always feasible (c=0 is a
// legal output), always > 0 (graph connected, n>=2). trivial.cpp reproduces it
// exactly -> ratio == 0.1 on every case.
//
// Score (min): sc = min(1000, 100*B / max(eps,F)); ratio = sc/1000.
// -----------------------------------------------------------------------------

typedef long double ld;

int n, m0, t, k, M;
vector<pair<int,int>> origEdges, candEdges;

// diagonal of the inverse of the t-grounded Laplacian, given a set of EXTRA
// edge indices (into candEdges) added on top of the original tree/graph.
ld worstResistanceTimesMass(const vector<int>& extraIdx, ld &massOut){
    int mm = m0 + (int)extraIdx.size();
    massOut = (ld)mm;
    int N = n - 1;
    vector<int> ridx(n + 1, -1);
    { int c = 0; for (int v = 1; v <= n; v++) if (v != t) ridx[v] = c++; }

    vector<vector<ld>> A(N, vector<ld>(N, 0.0L));
    auto addEdge = [&](int u, int v){
        if (u == t && v == t) return;
        if (u == t){ int rv = ridx[v]; A[rv][rv] += 1.0L; return; }
        if (v == t){ int ru = ridx[u]; A[ru][ru] += 1.0L; return; }
        int ru = ridx[u], rv = ridx[v];
        A[ru][ru] += 1.0L; A[rv][rv] += 1.0L;
        A[ru][rv] -= 1.0L; A[rv][ru] -= 1.0L;
    };
    for (auto &e : origEdges) addEdge(e.first, e.second);
    for (int idx : extraIdx) addEdge(candEdges[idx].first, candEdges[idx].second);

    // Gauss-Jordan with partial pivoting on [A | I] -> [I | A^{-1}]; N is small
    // (<= ~300), so this is comfortably fast and stable.
    vector<vector<ld>> I(N, vector<ld>(N, 0.0L));
    for (int i = 0; i < N; i++) I[i][i] = 1.0L;
    for (int col = 0; col < N; col++){
        int piv = col; ld best = fabsl(A[col][col]);
        for (int r = col + 1; r < N; r++){
            ld v = fabsl(A[r][col]);
            if (v > best){ best = v; piv = r; }
        }
        if (piv != col){ swap(A[piv], A[col]); swap(I[piv], I[col]); }
        ld d = A[col][col];
        if (fabsl(d) < 1e-12L) d = (d < 0 ? -1e-12L : 1e-12L);
        for (int j = 0; j < N; j++){ A[col][j] /= d; I[col][j] /= d; }
        for (int r = 0; r < N; r++){
            if (r == col) continue;
            ld factor = A[r][col];
            if (factor == 0.0L) continue;
            for (int j = 0; j < N; j++){ A[r][j] -= factor * A[col][j]; I[r][j] -= factor * I[col][j]; }
        }
    }
    ld worstR = 0.0L;
    for (int i = 0; i < N; i++) worstR = max(worstR, I[i][i]);
    return worstR;
}

double objective(const vector<int>& extraIdx){
    ld mass;
    ld R = worstResistanceTimesMass(extraIdx, mass);
    return (double)(2.0L * mass * R);
}

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    n = inf.readInt(); m0 = inf.readInt(); t = inf.readInt(); k = inf.readInt(); M = inf.readInt();
    origEdges.resize(m0);
    for (int i = 0; i < m0; i++){
        int u = inf.readInt(1, n, "u");
        int v = inf.readInt(1, n, "v");
        origEdges[i] = {u, v};
    }
    candEdges.resize(M);
    for (int i = 0; i < M; i++){
        int a = inf.readInt(1, n, "a");
        int b = inf.readInt(1, n, "b");
        candEdges[i] = {a, b};
    }

    int c = ouf.readInt(0, k, "c");
    vector<int> extraIdx; extraIdx.reserve(c);
    vector<char> used(M + 1, 0);
    for (int i = 0; i < c; i++){
        int idx = ouf.readInt(1, M, "idx");
        if (used[idx]) quitf(_wa, "candidate index %d chosen more than once", idx);
        used[idx] = 1;
        extraIdx.push_back(idx - 1);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens in output");

    double F = objective(extraIdx);
    double B = objective({});
    if (!isfinite(F) || !isfinite(B)) quitf(_wa, "non-finite objective");
    if (B <= 0.0) quitf(_wa, "internal error: non-positive baseline");

    double sc = min(1000.0, 100.0 * B / max(1e-9, F));
    quitp(sc / 1000.0, "OK F=%.6f B=%.6f c=%d Ratio: %.6f", F, B, c, sc / 1000.0);
    return 0;
}
