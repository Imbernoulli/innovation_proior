// TIER: greedy
// "Size for the dominant case." Builds the same stable backbone as trivial
// (chords + verticals + one diagonal/bay), then spends the remaining budget
// upgrading member cross-sections in decreasing order of how hard GRAVITY
// (load case 0 only) stresses them. This is the natural first instinct: solve
// the one case that clearly carries the most total load and size for it. It
// never looks at the other 15 cases, so it can leave members that matter only
// for lateral (wind) loads under-sized -- and gravity is symmetric & purely
// vertical, so it never even signals that some bays are structurally a
// mechanism if a diagonal were missing (they aren't missing here, but their
// AREA is chosen blind to wind/moving-load demand).
#include <bits/stdc++.h>
using namespace std;

static const double EPS_REG = 1e-3;
static const double DISP_CAP = 200.0; // mirrors the checker's mechanism detector

struct Truss {
    int N, M, K;
    double SPAN, H, E, SIGMA_Y, BUDGET;
    int NAREA, PIN, ROLLER;
    vector<double> AREA, X, Y;
    vector<int> U, V;
    vector<vector<int>> Lnode;
    vector<vector<double>> Lfx, Lfy;
};

Truss readInput() {
    Truss t;
    int W;
    cin >> W >> t.K;
    cin >> t.N >> t.M;
    cin >> t.SPAN >> t.H >> t.E >> t.SIGMA_Y;
    cin >> t.NAREA;
    t.AREA.resize(t.NAREA);
    for (auto &a : t.AREA) cin >> a;
    cin >> t.BUDGET;
    cin >> t.PIN >> t.ROLLER;
    t.X.resize(t.N); t.Y.resize(t.N);
    for (int i = 0; i < t.N; i++) cin >> t.X[i] >> t.Y[i];
    t.U.resize(t.M); t.V.resize(t.M);
    for (int i = 0; i < t.M; i++) cin >> t.U[i] >> t.V[i];
    int K2; cin >> K2; t.K = K2;
    t.Lnode.assign(K2, {}); t.Lfx.assign(K2, {}); t.Lfy.assign(K2, {});
    for (int c = 0; c < K2; c++) {
        int L; cin >> L;
        t.Lnode[c].resize(L); t.Lfx[c].resize(L); t.Lfy[c].resize(L);
        for (int j = 0; j < L; j++) cin >> t.Lnode[c][j] >> t.Lfx[c][j] >> t.Lfy[c][j];
    }
    return t;
}

double len2(const Truss &t, int u, int v) {
    double dx = t.X[v] - t.X[u], dy = t.Y[v] - t.Y[u];
    return sqrt(dx * dx + dy * dy);
}

// Solve one case for a given member set; fill per-member utilization (parallel to ids).
void solveCase(const Truss &t, const vector<int> &ids, const vector<double> &area,
               const vector<double> &lenv, int c, vector<double> &util) {
    int N = t.N, nd = 2 * N;
    vector<int> dofMap(nd, -1);
    int nf = 0;
    for (int i = 0; i < N; i++) {
        bool fixX = (i == t.PIN), fixY = (i == t.PIN || i == t.ROLLER);
        if (!fixX) dofMap[2 * i] = nf++;
        if (!fixY) dofMap[2 * i + 1] = nf++;
    }
    int Ssel = (int)ids.size();
    util.assign(Ssel, 0.0);
    if (nf == 0) return;
    vector<vector<double>> K(nf, vector<double>(nf, 0.0));
    vector<double> F(nf, 0.0);
    for (int m = 0; m < Ssel; m++) {
        int a = t.U[ids[m]], b = t.V[ids[m]];
        double L = lenv[m];
        if (L <= 1e-12) continue;
        double cx = (t.X[b] - t.X[a]) / L, sy = (t.Y[b] - t.Y[a]) / L;
        double k = t.E * area[m] / L;
        int dofs[4] = {2 * a, 2 * a + 1, 2 * b, 2 * b + 1};
        double dirs[4] = {-cx, -sy, cx, sy};
        for (int p = 0; p < 4; p++) {
            int gp = dofMap[dofs[p]]; if (gp < 0) continue;
            for (int q = 0; q < 4; q++) {
                int gq = dofMap[dofs[q]]; if (gq < 0) continue;
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
    for (int col = 0; col < nf; col++) {
        int piv = col; double best = fabs(K[col][col]);
        for (int r = col + 1; r < nf; r++) if (fabs(K[r][col]) > best) { best = fabs(K[r][col]); piv = r; }
        if (best < 1e-12) { util.assign(Ssel, 1e9); return; }
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
        if (fabs(K[i][i]) < 1e-12) { util.assign(Ssel, 1e9); return; }
        u[i] = s / K[i][i];
    }
    for (double v : u) if (!isfinite(v) || fabs(v) > DISP_CAP) { util.assign(Ssel, 1e9); return; }
    vector<double> ufull(nd, 0.0);
    for (int i = 0; i < N; i++) {
        int gx = dofMap[2 * i], gy = dofMap[2 * i + 1];
        ufull[2 * i] = (gx >= 0) ? u[gx] : 0.0;
        ufull[2 * i + 1] = (gy >= 0) ? u[gy] : 0.0;
    }
    for (int m = 0; m < Ssel; m++) {
        int a = t.U[ids[m]], b = t.V[ids[m]];
        double L = lenv[m];
        if (L <= 1e-12) continue;
        double cx = (t.X[b] - t.X[a]) / L, sy = (t.Y[b] - t.Y[a]) / L;
        double elong = cx * (ufull[2 * b] - ufull[2 * a]) + sy * (ufull[2 * b + 1] - ufull[2 * a + 1]);
        double stress = t.E * elong / L;
        util[m] = isfinite(stress) ? fabs(stress) / t.SIGMA_Y : 1e9;
    }
}

int main() {
    Truss t = readInput();
    int W = t.N / 2 - 1;
    vector<int> ids;
    for (int i = 0; i < W; i++) ids.push_back(i);              // bottom
    for (int i = 0; i < W; i++) ids.push_back(W + i);          // top
    for (int i = 0; i <= W; i++) ids.push_back(2 * W + i);     // vertical
    for (int i = 0; i < W; i++) ids.push_back(3 * W + 1 + i);  // diagonal A

    int nb = (int)ids.size();
    vector<int> areaIdx(nb, 0);
    vector<double> lenv(nb);
    for (int i = 0; i < nb; i++) lenv[i] = len2(t, t.U[ids[i]], t.V[ids[i]]);

    auto curCost = [&]() {
        double c = 0.0;
        for (int i = 0; i < nb; i++) c += lenv[i] * t.AREA[areaIdx[i]];
        return c;
    };
    double costMin = curCost();
    double remaining = t.BUDGET - costMin;

    vector<double> area0(nb);
    for (int i = 0; i < nb; i++) area0[i] = t.AREA[0];
    vector<double> util0;
    solveCase(t, ids, area0, lenv, 0, util0); // gravity case only

    vector<int> order(nb);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b) { return util0[a] > util0[b]; });

    for (int oi = 0; oi < nb; oi++) {
        int i = order[oi];
        while (areaIdx[i] < t.NAREA - 1) {
            double step = (t.AREA[areaIdx[i] + 1] - t.AREA[areaIdx[i]]) * lenv[i];
            if (remaining + 1e-9 < step) break;
            remaining -= step; areaIdx[i]++;
        }
    }

    printf("%d\n", nb);
    for (int i = 0; i < nb; i++) printf("%d %d\n", ids[i], areaIdx[i]);
    return 0;
}
