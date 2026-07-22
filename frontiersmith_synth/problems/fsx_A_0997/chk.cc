#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Sixteen-Load Truss Margin"  (family multi-load-truss-margin)
//
// Solves real 2D pin-jointed truss statics (direct stiffness method, small dense
// Gaussian elimination with a tiny Tikhonov regularization so a kinematic
// mechanism never divides by an exact zero -- it just blows the resulting stress
// up, which our margin clamp turns into 0 for that load case, exactly modelling
// "unstable under this load").
//
// Objective (MAX) F = min over the K load cases of margin_c = clamp(1 - max
// member utilization in that case, 0, 1).  Baseline B = the SAME quantity for a
// fixed, weak, but always-STABLE reference: chords + verticals + ONE diagonal
// per bay (type A only) at the SMALLEST area class -- a minimal, fully
// triangulated (hence never a mechanism, hence never spuriously 0) but
// thin-margin design.  ratio = min(1, 0.1*F/B), so a submission that reproduces
// the reference scores ~0.1 and a submission ~9x safer scores near the 1.0 cap.
// -----------------------------------------------------------------------------

static const double EPS_REG = 1e-3;
// A kinematic mechanism (e.g. a bay with no diagonal) can be mechanically
// DECOUPLED from a particular load direction, so real member stresses can stay
// near zero even while the regularized solve produces a huge displacement in
// that direction (the load is entirely absorbed by the tiny EPS_REG "ghost"
// stiffness). Stress alone would then miss the instability. A displacement
// this large (>100x anything a genuinely stable, near-yield structure produces
// in this geometry) is the signature of exactly that -- flag it as unstable.
static const double DISP_CAP = 200.0;

struct Truss {
    int N, M;
    double SPAN, H, E, SIGMA_Y;
    int NAREA;
    vector<double> AREA;
    double BUDGET;
    int PIN, ROLLER;
    vector<double> X, Y;
    vector<int> U, V; // candidate members, 0-indexed
    int K;
    vector<vector<int>> Lnode;
    vector<vector<double>> Lfx, Lfy;
};

Truss readInput() {
    Truss t;
    t.SPAN = t.H = t.E = t.SIGMA_Y = 0;
    int W;
    W = inf.readInt();
    t.K = inf.readInt();
    t.N = inf.readInt();
    t.M = inf.readInt();
    t.SPAN = inf.readDouble();
    t.H = inf.readDouble();
    t.E = inf.readDouble();
    t.SIGMA_Y = inf.readDouble();
    t.NAREA = inf.readInt();
    t.AREA.resize(t.NAREA);
    for (int i = 0; i < t.NAREA; i++) t.AREA[i] = inf.readDouble();
    t.BUDGET = inf.readDouble();
    t.PIN = inf.readInt();
    t.ROLLER = inf.readInt();
    t.X.resize(t.N); t.Y.resize(t.N);
    for (int i = 0; i < t.N; i++) { t.X[i] = inf.readDouble(); t.Y[i] = inf.readDouble(); }
    t.U.resize(t.M); t.V.resize(t.M);
    for (int i = 0; i < t.M; i++) { t.U[i] = inf.readInt(); t.V[i] = inf.readInt(); }
    int K2 = inf.readInt();
    (void)W;
    t.Lnode.assign(K2, {}); t.Lfx.assign(K2, {}); t.Lfy.assign(K2, {});
    for (int c = 0; c < K2; c++) {
        int L = inf.readInt();
        t.Lnode[c].resize(L); t.Lfx[c].resize(L); t.Lfy[c].resize(L);
        for (int j = 0; j < L; j++) {
            t.Lnode[c][j] = inf.readInt();
            t.Lfx[c][j] = inf.readDouble();
            t.Lfy[c][j] = inf.readDouble();
        }
    }
    t.K = K2;
    return t;
}

double memberLength(const Truss &t, int u, int v) {
    double dx = t.X[v] - t.X[u], dy = t.Y[v] - t.Y[u];
    return sqrt(dx * dx + dy * dy);
}

// Solve one load case for a fixed selection of members (endpoints + chosen area).
// Returns the per-case margin in [0,1] (0 if unstable / overstressed / non-finite).
double solveCaseMargin(const Truss &t, const vector<int> &selU, const vector<int> &selV,
                        const vector<double> &selArea, const vector<double> &selLen,
                        int c) {
    int N = t.N;
    int nd = 2 * N;
    vector<int> dofMap(nd, -1);
    int nf = 0;
    for (int i = 0; i < N; i++) {
        bool fixX = (i == t.PIN);
        bool fixY = (i == t.PIN || i == t.ROLLER);
        if (!fixX) dofMap[2 * i] = nf++;
        if (!fixY) dofMap[2 * i + 1] = nf++;
    }
    if (nf == 0) return 0.0;
    vector<vector<double>> K(nf, vector<double>(nf, 0.0));
    vector<double> F(nf, 0.0);
    int Ssel = (int)selU.size();
    for (int m = 0; m < Ssel; m++) {
        int a = selU[m], b = selV[m];
        double L = selLen[m];
        if (L <= 1e-12) continue;
        double cx = (t.X[b] - t.X[a]) / L, sy = (t.Y[b] - t.Y[a]) / L;
        double k = t.E * selArea[m] / L;
        int dofs[4] = {2 * a, 2 * a + 1, 2 * b, 2 * b + 1};
        double dirs[4] = {-cx, -sy, cx, sy};
        for (int p = 0; p < 4; p++) {
            int gp = dofMap[dofs[p]];
            if (gp < 0) continue;
            for (int q = 0; q < 4; q++) {
                int gq = dofMap[dofs[q]];
                if (gq < 0) continue;
                K[gp][gq] += k * dirs[p] * dirs[q];
            }
        }
    }
    for (int j = 0; j < (int)t.Lnode[c].size(); j++) {
        int nodeId = t.Lnode[c][j];
        int gx = dofMap[2 * nodeId], gy = dofMap[2 * nodeId + 1];
        if (gx >= 0) F[gx] += t.Lfx[c][j];
        if (gy >= 0) F[gy] += t.Lfy[c][j];
    }
    for (int i = 0; i < nf; i++) K[i][i] += EPS_REG;

    // Gaussian elimination with partial pivoting.
    for (int col = 0; col < nf; col++) {
        int piv = col;
        double best = fabs(K[col][col]);
        for (int r = col + 1; r < nf; r++) {
            if (fabs(K[r][col]) > best) { best = fabs(K[r][col]); piv = r; }
        }
        if (best < 1e-12) return 0.0; // numerically singular even after regularization
        if (piv != col) { swap(K[piv], K[col]); swap(F[piv], F[col]); }
        double pv = K[col][col];
        for (int r = col + 1; r < nf; r++) {
            double factor = K[r][col] / pv;
            if (factor == 0.0) continue;
            for (int cc = col; cc < nf; cc++) K[r][cc] -= factor * K[col][cc];
            F[r] -= factor * F[col];
        }
    }
    vector<double> u(nf, 0.0);
    for (int i = nf - 1; i >= 0; i--) {
        double s = F[i];
        for (int j = i + 1; j < nf; j++) s -= K[i][j] * u[j];
        if (fabs(K[i][i]) < 1e-12) return 0.0;
        u[i] = s / K[i][i];
    }
    for (double v : u) if (!isfinite(v) || fabs(v) > DISP_CAP) return 0.0; // mechanism, this load excites it
    vector<double> ufull(nd, 0.0);
    for (int i = 0; i < N; i++) {
        int gx = dofMap[2 * i], gy = dofMap[2 * i + 1];
        ufull[2 * i] = (gx >= 0) ? u[gx] : 0.0;
        ufull[2 * i + 1] = (gy >= 0) ? u[gy] : 0.0;
    }
    double maxUtil = 0.0;
    for (int m = 0; m < Ssel; m++) {
        int a = selU[m], b = selV[m];
        double L = selLen[m];
        if (L <= 1e-12) continue;
        double cx = (t.X[b] - t.X[a]) / L, sy = (t.Y[b] - t.Y[a]) / L;
        double elong = cx * (ufull[2 * b] - ufull[2 * a]) + sy * (ufull[2 * b + 1] - ufull[2 * a + 1]);
        double stress = t.E * elong / L; // force/area = E*elong/L (area-independent)
        if (!isfinite(stress)) return 0.0;
        double util = fabs(stress) / t.SIGMA_Y;
        if (util > maxUtil) maxUtil = util;
    }
    if (!isfinite(maxUtil)) return 0.0;
    double margin = 1.0 - maxUtil;
    if (margin < 0.0) margin = 0.0;
    if (margin > 1.0) margin = 1.0;
    return margin;
}

double worstCaseMargin(const Truss &t, const vector<int> &selU, const vector<int> &selV,
                        const vector<double> &selArea, const vector<double> &selLen) {
    double worst = 1.0;
    for (int c = 0; c < t.K; c++) {
        double m = solveCaseMargin(t, selU, selV, selArea, selLen, c);
        if (m < worst) worst = m;
        if (worst <= 0.0) return 0.0;
    }
    return worst;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);
    Truss t = readInput();

    // ---- internal baseline B: chords + verticals + ONE diagonal/bay (type A),
    //      smallest area class for every member -- weak but always determinate. ----
    {
        // Recover W from N: N = 2*(W+1)
        int W = t.N / 2 - 1;
        vector<int> bu, bv; vector<double> ba, bl;
        auto addMem = [&](int a, int b) {
            bu.push_back(a); bv.push_back(b);
            ba.push_back(t.AREA[0]);
            bl.push_back(memberLength(t, a, b));
        };
        for (int i = 0; i < W; i++) addMem(i, i + 1);                 // bottom chord
        for (int i = 0; i < W; i++) addMem((W + 1) + i, (W + 1) + i + 1); // top chord
        for (int i = 0; i <= W; i++) addMem(i, (W + 1) + i);          // vertical
        for (int i = 0; i < W; i++) addMem(i, (W + 1) + i + 1);       // diagonal A
        double B = worstCaseMargin(t, bu, bv, ba, bl);
        if (B <= 1e-9) B = 1e-9; // generator is calibrated so this shouldn't trigger

        // ---- participant's selection ----
        int S = ouf.readInt(0, t.M, "S");
        vector<int> su(S), sv(S); vector<double> sa(S), sl(S);
        vector<char> used(t.M, 0);
        double cost = 0.0;
        for (int i = 0; i < S; i++) {
            int mid = ouf.readInt(0, t.M - 1, "member_idx");
            if (used[mid]) quitf(_wa, "member %d selected more than once", mid);
            used[mid] = 1;
            int aidx = ouf.readInt(0, t.NAREA - 1, "area_idx");
            su[i] = t.U[mid]; sv[i] = t.V[mid];
            sa[i] = t.AREA[aidx];
            sl[i] = memberLength(t, t.U[mid], t.V[mid]);
            cost += sl[i] * sa[i];
        }
        if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");
        if (cost > t.BUDGET * (1.0 + 1e-6) + 1e-9)
            quitf(_wa, "budget exceeded: cost=%.6f > BUDGET=%.6f", cost, t.BUDGET);

        double F = worstCaseMargin(t, su, sv, sa, sl);
        double sc = min(1000.0, 100.0 * F / max(1e-9, B));
        quitp(sc / 1000.0, "OK F=%.6f B=%.6f cost=%.6f/%.6f Ratio: %.6f",
              F, B, cost, t.BUDGET, sc / 1000.0);
    }
    return 0;
}
