// TIER: greedy
// The obvious first attempt: a fully-stressed-design style rule. Solve the
// structure once at minimum area, rank bars by the MAGNITUDE of their own
// internal axial force, and hand out catalog tiers in proportion to that force
// rank (highest-force bar gets the biggest/most expensive section, lowest-force
// bar stays cheapest). This sizes every bar purely from its own local force,
// ignoring (a) that a short bar can carry high force yet contribute almost
// nothing to the *target* deflection (its elongation is tiny regardless of
// area), and (b) that the target is a specific nonzero deflection, not "as
// stiff as possible" -- always favoring the highest-force members monotonically
// drives the structure stiffer, which can overshoot a mid-range target.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static int N, M, K;
static vector<double> X, Y;
static vector<int> SUP;
static vector<int> BA, BB;
static vector<ll> AREA, COST;
static double E;
static int Lc;
static vector<int> LNODE;
static vector<double> LFX, LFY;

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

int main() {
    scanf("%d %d %d", &N, &M, &K);
    X.resize(N); Y.resize(N); SUP.resize(N);
    for (int i = 0; i < N; i++) { ll x, y; int s; scanf("%lld %lld %d", &x, &y, &s); X[i] = x; Y[i] = y; SUP[i] = s; }
    BA.resize(M); BB.resize(M);
    for (int i = 0; i < M; i++) { int a, b; scanf("%d %d", &a, &b); BA[i] = a - 1; BB[i] = b - 1; }
    AREA.resize(K); COST.resize(K);
    for (int k = 0; k < K; k++) { ll a, c; scanf("%lld %lld", &a, &c); AREA[k] = a; COST[k] = c; }
    ll El; scanf("%lld", &El); E = (double)El;
    ll Wd, Wc; scanf("%lld %lld", &Wd, &Wc);
    scanf("%d", &Lc);
    LNODE.resize(Lc); LFX.resize(Lc); LFY.resize(Lc);
    for (int j = 0; j < Lc; j++) {
        int nd; ll fx, fy, tx, ty;
        scanf("%d %lld %lld %lld %lld", &nd, &fx, &fy, &tx, &ty);
        LNODE[j] = nd - 1; LFX[j] = (double)fx; LFY[j] = (double)fy;
    }

    freeIdx.assign(2 * N, -1);
    NFREE = 0;
    for (int i = 0; i < N; i++) {
        bool fixX = (SUP[i] == 1);
        bool fixY = (SUP[i] == 1 || SUP[i] == 2);
        if (!fixX) freeIdx[2 * i] = NFREE++;
        if (!fixY) freeIdx[2 * i + 1] = NFREE++;
    }

    // baseline solve at minimum area for all bars -> per-bar internal force
    vector<vector<double>> Kf(NFREE, vector<double>(NFREE, 0.0));
    vector<double> Ff(NFREE, 0.0);
    vector<double> len(M), cc(M), ss(M);
    for (int i = 0; i < M; i++) {
        int a = BA[i], b = BB[i];
        double dxv = X[b] - X[a], dyv = Y[b] - Y[a];
        double Lb = sqrt(dxv * dxv + dyv * dyv);
        len[i] = Lb;
        double c = dxv / Lb, s = dyv / Lb;
        cc[i] = c; ss[i] = s;
        double k = E * (double)AREA[0] / Lb;
        int dofs[4] = {2 * a, 2 * a + 1, 2 * b, 2 * b + 1};
        double cx = c * c, sx = s * s, cs = c * s;
        double km[4][4] = {{cx, cs, -cx, -cs}, {cs, sx, -cs, -sx}, {-cx, -cs, cx, cs}, {-cs, -sx, cs, sx}};
        for (int p = 0; p < 4; p++) {
            int fp = freeIdx[dofs[p]]; if (fp < 0) continue;
            for (int q = 0; q < 4; q++) { int fq = freeIdx[dofs[q]]; if (fq < 0) continue; Kf[fp][fq] += k * km[p][q]; }
        }
    }
    for (int j = 0; j < Lc; j++) {
        int nd = LNODE[j];
        int fx = freeIdx[2 * nd], fy = freeIdx[2 * nd + 1];
        if (fx >= 0) Ff[fx] += LFX[j];
        if (fy >= 0) Ff[fy] += LFY[j];
    }
    vector<double> u = gaussSolve(Kf, Ff);

    vector<double> force(M);
    for (int i = 0; i < M; i++) {
        int a = BA[i], b = BB[i];
        int fxA = freeIdx[2 * a], fyA = freeIdx[2 * a + 1], fxB = freeIdx[2 * b], fyB = freeIdx[2 * b + 1];
        double uxA = fxA >= 0 ? u[fxA] : 0.0, uyA = fyA >= 0 ? u[fyA] : 0.0;
        double uxB = fxB >= 0 ? u[fxB] : 0.0, uyB = fyB >= 0 ? u[fyB] : 0.0;
        double elong = (uxB - uxA) * cc[i] + (uyB - uyA) * ss[i];
        double stiffness = E * (double)AREA[0] / len[i];
        force[i] = fabs(stiffness * elong);
    }

    vector<int> order(M);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b) { return force[a] > force[b]; });

    vector<int> idx(M, 0);
    for (int r = 0; r < M; r++) {
        int bar = order[r];
        int tier = (M > 1) ? (int)llround((double)(M - 1 - r) / (double)(M - 1) * (K - 1)) : (K - 1);
        idx[bar] = tier;
    }

    for (int i = 0; i < M; i++) printf("%d\n", idx[i]);
    return 0;
}
