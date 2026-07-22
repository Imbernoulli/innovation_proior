// TIER: strong
// Insight: displacement is a GLOBAL response u = K^{-1} F; a bar's own force
// says nothing about how much upgrading it moves u toward the target. What
// matters is its baseline STRAIN ENERGY e_i = (E*A0/L_i) * elongation_i^2 --
// this is the standard compliance-sensitivity proxy: bars that already store a
// lot of elastic energy under the baseline load sit on the dominant load path
// and are the ones whose stiffness controls the global response; a short bar
// pinned to a load point can carry high force yet have tiny elongation (hence
// tiny e_i) because it is already effectively rigid at any catalog area, so
// enlarging it wastes cost. We rank bars once by e_i (ONE baseline solve),
// then search how many of the TOP-ranked bars to upgrade to the top catalog
// tier (a handful of full re-solves to actually measure the true objective,
// since the right amount of stiffening depends on the target, not just "more"),
// and finally locally refine the tier of just that dominant subset.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

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

static vector<int> freeIdx;
static int NFREE;

static vector<double> gaussSolve(vector<vector<double>> A, vector<double> b) {
    int n = (int)A.size();
    for (int col = 0; col < n; col++) {
        int piv = col; double best = fabs(A[col][col]);
        for (int r = col + 1; r < n; r++) { double v = fabs(A[r][col]); if (v > best) { best = v; piv = r; } }
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

struct FemResult { double dispErr, cost; vector<double> elong; };

static FemResult solveFEM(const vector<int>& idx, bool wantElong) {
    vector<vector<double>> Kf(NFREE, vector<double>(NFREE, 0.0));
    vector<double> Ff(NFREE, 0.0);
    vector<double> len(M), cc(M), ss(M);
    double totalCost = 0.0;
    for (int i = 0; i < M; i++) {
        int a = BA[i], b = BB[i];
        double dxv = X[b] - X[a], dyv = Y[b] - Y[a];
        double Lb = sqrt(dxv * dxv + dyv * dyv);
        len[i] = Lb;
        double c = dxv / Lb, s = dyv / Lb;
        cc[i] = c; ss[i] = s;
        ll Ai = AREA[idx[i]];
        double k = E * (double)Ai / Lb;
        int dofs[4] = {2 * a, 2 * a + 1, 2 * b, 2 * b + 1};
        double cx = c * c, sx = s * s, cs = c * s;
        double km[4][4] = {{cx, cs, -cx, -cs}, {cs, sx, -cs, -sx}, {-cx, -cs, cx, cs}, {-cs, -sx, cs, sx}};
        for (int p = 0; p < 4; p++) {
            int fp = freeIdx[dofs[p]]; if (fp < 0) continue;
            for (int q = 0; q < 4; q++) { int fq = freeIdx[dofs[q]]; if (fq < 0) continue; Kf[fp][fq] += k * km[p][q]; }
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
    FemResult res;
    res.dispErr = sqrt(errsq);
    res.cost = totalCost;
    if (wantElong) {
        res.elong.resize(M);
        for (int i = 0; i < M; i++) {
            int a = BA[i], b = BB[i];
            int fxA = freeIdx[2 * a], fyA = freeIdx[2 * a + 1], fxB = freeIdx[2 * b], fyB = freeIdx[2 * b + 1];
            double uxA = fxA >= 0 ? u[fxA] : 0.0, uyA = fyA >= 0 ? u[fyA] : 0.0;
            double uxB = fxB >= 0 ? u[fxB] : 0.0, uyB = fyB >= 0 ? u[fyB] : 0.0;
            res.elong[i] = (uxB - uxA) * cc[i] + (uyB - uyA) * ss[i];
        }
    }
    return res;
}

static double objective(const FemResult& r) { return WDISP * r.dispErr + WCOST * r.cost; }

int main() {
    scanf("%d %d %d", &N, &M, &K);
    X.resize(N); Y.resize(N); SUP.resize(N);
    for (int i = 0; i < N; i++) { ll x, y; int s; scanf("%lld %lld %d", &x, &y, &s); X[i] = x; Y[i] = y; SUP[i] = s; }
    BA.resize(M); BB.resize(M);
    for (int i = 0; i < M; i++) { int a, b; scanf("%d %d", &a, &b); BA[i] = a - 1; BB[i] = b - 1; }
    AREA.resize(K); COST.resize(K);
    for (int k = 0; k < K; k++) { ll a, c; scanf("%lld %lld", &a, &c); AREA[k] = a; COST[k] = c; }
    ll El; scanf("%lld", &El); E = (double)El;
    ll Wd, Wc; scanf("%lld %lld", &Wd, &Wc); WDISP = (double)Wd; WCOST = (double)Wc;
    scanf("%d", &Lc);
    LNODE.resize(Lc); LFX.resize(Lc); LFY.resize(Lc); LTX.resize(Lc); LTY.resize(Lc);
    for (int j = 0; j < Lc; j++) {
        int nd; ll fx, fy, tx, ty;
        scanf("%d %lld %lld %lld %lld", &nd, &fx, &fy, &tx, &ty);
        LNODE[j] = nd - 1; LFX[j] = (double)fx; LFY[j] = (double)fy; LTX[j] = (double)tx; LTY[j] = (double)ty;
    }

    freeIdx.assign(2 * N, -1);
    NFREE = 0;
    for (int i = 0; i < N; i++) {
        bool fixX = (SUP[i] == 1);
        bool fixY = (SUP[i] == 1 || SUP[i] == 2);
        if (!fixX) freeIdx[2 * i] = NFREE++;
        if (!fixY) freeIdx[2 * i + 1] = NFREE++;
    }

    vector<int> idxBase(M, 0);
    FemResult base = solveFEM(idxBase, true);

    // strain-energy sensitivity proxy per bar (dominant load-path score)
    vector<double> energy(M);
    for (int i = 0; i < M; i++) {
        int a = BA[i], b = BB[i];
        double dxv = X[b] - X[a], dyv = Y[b] - Y[a];
        double Lb = sqrt(dxv * dxv + dyv * dyv);
        double k0 = E * (double)AREA[0] / Lb;
        energy[i] = k0 * base.elong[i] * base.elong[i];
    }
    vector<int> order(M);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b) { return energy[a] > energy[b]; });

    double fracs[] = {0.0, 0.02, 0.04, 0.06, 0.09, 0.13, 0.18, 0.25, 0.35, 0.5, 0.7, 1.0};
    int floorTiers[] = {0, 1};
    vector<int> best(M, 0);
    double bestF = objective(base);
    for (int floorT : floorTiers) {
        for (double f : fracs) {
            if (floorT != 0 && f == 0.0) continue; // same as an all-floor uniform case below
            int cnt = (int)llround(f * M);
            vector<int> cand(M, floorT);
            for (int r = 0; r < cnt; r++) cand[order[r]] = K - 1;
            FemResult fr = solveFEM(cand, false);
            double val = objective(fr);
            if (val < bestF) { bestF = val; best = cand; }
        }
    }
    // broad/uniform stiffening candidates (subsumes a "spread evenly" strategy)
    for (int t = 0; t < K; t++) {
        vector<int> cand(M, t);
        FemResult fr = solveFEM(cand, false);
        double val = objective(fr);
        if (val < bestF) { bestF = val; best = cand; }
    }
    // graded ramp by ENERGY rank (same shape idea as a force-proportional rule,
    // but driven by the correct global-sensitivity signal instead of raw force)
    {
        vector<int> ramp(M, 0);
        for (int r = 0; r < M; r++) {
            int bar = order[r];
            int tier = (M > 1) ? (int)llround((double)(M - 1 - r) / (double)(M - 1) * (K - 1)) : (K - 1);
            ramp[bar] = tier;
        }
        FemResult fr = solveFEM(ramp, false);
        double val = objective(fr);
        if (val < bestF) { bestF = val; best = ramp; }
    }

    // local refinement: try each tier for bars in the top ~30% by energy
    int topCnt = max(1, (int)llround(0.3 * M));
    for (int pass = 0; pass < 3; pass++) {
        for (int r = 0; r < topCnt; r++) {
            int bar = order[r];
            int origTier = best[bar];
            int bestTier = origTier;
            double bestVal = bestF;
            for (int t = 0; t < K; t++) {
                if (t == origTier) continue;
                best[bar] = t;
                FemResult fr = solveFEM(best, false);
                double val = objective(fr);
                if (val < bestVal) { bestVal = val; bestTier = t; }
            }
            best[bar] = bestTier;
            bestF = bestVal;
        }
    }

    for (int i = 0; i < M; i++) printf("%d\n", best[i]);
    return 0;
}
