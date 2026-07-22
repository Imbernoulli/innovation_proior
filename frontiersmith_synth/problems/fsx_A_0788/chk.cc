#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker for "Size Truss Members for a Target Deflection".
// Reads a fixed truss (nodes/bars/supports), external loads + a target
// displacement at each loaded node, and a discrete cross-section catalog.
// Participant output: one catalog index (0..K-1) per bar, in bar input order.
// Assembles the global stiffness matrix K (2D truss elements), solves K*u=F for
// the free-DOF displacements via dense Gaussian elimination, and scores
//   F = Wdisp * ||u_loaded - target||_2 + Wcost * sum(cost_k(bar) * length(bar))
// (minimize). Baseline B = same formula with every bar at catalog index 0
// (cheapest/most flexible = the "do nothing" reference).
// -----------------------------------------------------------------------------

static int N, M, K;
static vector<double> X, Y;
static vector<int> SUP;
static vector<int> BA, BB;
static vector<ll> AREA, COST;
static double E;
static double WDISP, WCOST;
static int Lc;
static vector<int> LNODE;
static vector<double> LFX, LFY, LTX, LTY;

static vector<int> freeIdx; // size 2N, -1 if fixed else compact free-dof index
static int NFREE;

static vector<double> gaussSolve(vector<vector<double>> A, vector<double> b) {
    int n = (int)A.size();
    for (int col = 0; col < n; col++) {
        int piv = col; double best = fabs(A[col][col]);
        for (int r = col + 1; r < n; r++) {
            double v = fabs(A[r][col]);
            if (v > best) { best = v; piv = r; }
        }
        if (piv != col) { swap(A[col], A[piv]); swap(b[col], b[piv]); }
        double diag = A[col][col];
        if (fabs(diag) < 1e-12) diag = (diag >= 0 ? 1e-12 : -1e-12);
        for (int r = col + 1; r < n; r++) {
            double f = A[r][col] / diag;
            if (f == 0.0) continue;
            for (int c2 = col; c2 < n; c2++) A[r][c2] -= f * A[col][c2];
            b[r] -= f * b[col];
        }
    }
    vector<double> x(n, 0.0);
    for (int r = n - 1; r >= 0; r--) {
        double s = b[r];
        for (int c2 = r + 1; c2 < n; c2++) s -= A[r][c2] * x[c2];
        double diag = A[r][r];
        if (fabs(diag) < 1e-12) diag = (diag >= 0 ? 1e-12 : -1e-12);
        x[r] = s / diag;
    }
    return x;
}

// Computes F = Wdisp*dispErr + Wcost*totalCost for a given per-bar catalog choice.
static double evalObjective(const vector<int>& idx) {
    vector<vector<double>> Kf(NFREE, vector<double>(NFREE, 0.0));
    vector<double> Ff(NFREE, 0.0);
    double totalCost = 0.0;
    for (int i = 0; i < M; i++) {
        int a = BA[i], b = BB[i];
        double dxv = X[b] - X[a], dyv = Y[b] - Y[a];
        double Lb = sqrt(dxv * dxv + dyv * dyv);
        double c = dxv / Lb, s = dyv / Lb;
        ll Ai = AREA[idx[i]];
        double k = E * (double)Ai / Lb;
        int dofs[4] = {2 * a, 2 * a + 1, 2 * b, 2 * b + 1};
        double cx = c * c, sx = s * s, cs = c * s;
        double km[4][4] = {
            { cx,  cs, -cx, -cs},
            { cs,  sx, -cs, -sx},
            {-cx, -cs,  cx,  cs},
            {-cs, -sx,  cs,  sx}
        };
        for (int p = 0; p < 4; p++) {
            int fp = freeIdx[dofs[p]];
            if (fp < 0) continue;
            for (int q = 0; q < 4; q++) {
                int fq = freeIdx[dofs[q]];
                if (fq < 0) continue;
                Kf[fp][fq] += k * km[p][q];
            }
        }
        totalCost += (double)COST[idx[i]] * Lb;
    }
    for (int j = 0; j < Lc; j++) {
        int nd = LNODE[j];
        int fx = freeIdx[2 * nd], fy = freeIdx[2 * nd + 1];
        if (fx >= 0) Ff[fx] += LFX[j];
        if (fy >= 0) Ff[fy] += LFY[j];
    }
    vector<double> u = gaussSolve(Kf, Ff);
    double errsq = 0.0;
    for (int j = 0; j < Lc; j++) {
        int nd = LNODE[j];
        int fx = freeIdx[2 * nd], fy = freeIdx[2 * nd + 1];
        double ux = fx >= 0 ? u[fx] : 0.0;
        double uy = fy >= 0 ? u[fy] : 0.0;
        double ex = ux - LTX[j], ey = uy - LTY[j];
        errsq += ex * ex + ey * ey;
    }
    double dispErr = sqrt(errsq);
    return WDISP * dispErr + WCOST * totalCost;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    N = inf.readInt(); M = inf.readInt(); K = inf.readInt();
    X.resize(N); Y.resize(N); SUP.resize(N);
    for (int i = 0; i < N; i++) {
        X[i] = (double)inf.readLong();
        Y[i] = (double)inf.readLong();
        SUP[i] = inf.readInt(0, 2);
    }
    BA.resize(M); BB.resize(M);
    for (int i = 0; i < M; i++) {
        BA[i] = inf.readInt(1, N) - 1;
        BB[i] = inf.readInt(1, N) - 1;
    }
    AREA.resize(K); COST.resize(K);
    for (int k = 0; k < K; k++) {
        AREA[k] = inf.readLong();
        COST[k] = inf.readLong();
    }
    E = (double)inf.readLong();
    WDISP = (double)inf.readLong();
    WCOST = (double)inf.readLong();
    Lc = inf.readInt();
    LNODE.resize(Lc); LFX.resize(Lc); LFY.resize(Lc); LTX.resize(Lc); LTY.resize(Lc);
    for (int j = 0; j < Lc; j++) {
        LNODE[j] = inf.readInt(1, N) - 1;
        LFX[j] = (double)inf.readLong();
        LFY[j] = (double)inf.readLong();
        LTX[j] = (double)inf.readLong();
        LTY[j] = (double)inf.readLong();
    }

    freeIdx.assign(2 * N, -1);
    NFREE = 0;
    for (int i = 0; i < N; i++) {
        bool fixX = (SUP[i] == 1);
        bool fixY = (SUP[i] == 1 || SUP[i] == 2);
        if (!fixX) freeIdx[2 * i] = NFREE++;
        if (!fixY) freeIdx[2 * i + 1] = NFREE++;
    }

    // ---- read & validate participant output: M catalog indices in [0,K-1] ----
    vector<int> idx(M);
    for (int i = 0; i < M; i++) idx[i] = ouf.readInt(0, K - 1, "catalog index");
    if (!ouf.seekEof()) quitf(_wa, "trailing output after %d catalog indices", M);

    double F = evalObjective(idx);
    if (!isfinite(F)) quitf(_wa, "non-finite objective");

    vector<int> idxBase(M, 0);
    double B = evalObjective(idxBase);
    if (B < 1e-6) B = 1e-6;

    double sc = min(1000.0, 100.0 * B / max(1e-6, F));
    quitp(sc / 1000.0, "OK F=%.6f B=%.6f Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
