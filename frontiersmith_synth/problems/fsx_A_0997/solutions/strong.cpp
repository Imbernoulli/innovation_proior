// TIER: strong
// Insight: analyze the WHOLE 16-case sweep, not the dominant case. Starting
// from the same stable backbone as greedy, for every currently-missing second
// diagonal (the "X-bracing" candidate in each bay) we directly test its
// marginal effect on the worst-case-over-16-cases margin, and add the ones
// that help most per unit budget first -- this is exactly how a member that
// sits at the intersection of MANY load paths (serves gravity shear AND both
// wind directions AND several moving-load positions in the same bay) gets
// found: it is the one whose addition raises the *worst* case, not just some
// case. Remaining budget is then spent upgrading cross-sections, ranked by
// each member's WORST utilization across all 16 cases (not just gravity) --
// so a member that is merely warm under gravity but critical under wind still
// gets reinforced, which is exactly what the gravity-only greedy misses.
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

double worstMargin(const Truss &t, const vector<int> &ids, const vector<double> &area, const vector<double> &lenv) {
    double worst = 1.0;
    vector<double> util;
    for (int c = 0; c < t.K; c++) {
        solveCase(t, ids, area, lenv, c, util);
        double mx = 0.0;
        for (double u : util) mx = max(mx, u);
        double margin = max(0.0, min(1.0, 1.0 - mx));
        worst = min(worst, margin);
        if (worst <= 0.0) return 0.0;
    }
    return worst;
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
    for (int i = 0; i < W; i++) ids.push_back(4 * W + 1 + i);  // diagonal B (candidates, not yet selected)
    int M = (int)ids.size();

    vector<char> inSel(M, 0);
    for (int i = 0; i < nb; i++) inSel[i] = 1;
    vector<int> areaIdx(M, 0);
    vector<double> lenv(M);
    for (int i = 0; i < M; i++) lenv[i] = len2(t, t.U[ids[i]], t.V[ids[i]]);

    auto buildSel = [&](vector<int> &selIds, vector<double> &selArea, vector<double> &selLen) {
        selIds.clear(); selArea.clear(); selLen.clear();
        for (int i = 0; i < M; i++) if (inSel[i]) {
            selIds.push_back(ids[i]);
            selArea.push_back(t.AREA[areaIdx[i]]);
            selLen.push_back(lenv[i]);
        }
    };

    vector<int> selIds; vector<double> selArea, selLen;
    buildSel(selIds, selArea, selLen);
    double costMin = 0.0;
    for (int i = 0; i < nb; i++) costMin += lenv[i] * t.AREA[0];
    double remaining = t.BUDGET - costMin;

    double Fback = worstMargin(t, selIds, selArea, selLen);

    // Stage 1: test each candidate diagonal-B addition's marginal benefit, add the
    // best-benefit-per-cost ones first while affordable.
    vector<pair<double,int>> benefit; // (benefit, local index into [nb..M))
    for (int i = nb; i < M; i++) {
        vector<int> trialIds = selIds; trialIds.push_back(ids[i]);
        vector<double> trialArea = selArea; trialArea.push_back(t.AREA[0]);
        vector<double> trialLen = selLen; trialLen.push_back(lenv[i]);
        double Ftrial = worstMargin(t, trialIds, trialArea, trialLen);
        benefit.push_back({Ftrial - Fback, i});
    }
    sort(benefit.begin(), benefit.end(), [](const pair<double,int> &a, const pair<double,int> &b) {
        return a.first > b.first;
    });
    for (auto &pr : benefit) {
        if (pr.first <= 1e-9) continue;
        int i = pr.second;
        double c_ = lenv[i] * t.AREA[0];
        if (remaining + 1e-9 >= c_) { inSel[i] = 1; areaIdx[i] = 0; remaining -= c_; }
    }

    // Stage 2: rank currently-selected members by their WORST utilization across
    // all 16 cases (computed once on the current structure), upgrade area classes
    // greedily while budget remains.
    buildSel(selIds, selArea, selLen);
    vector<int> selPos;
    for (int i = 0; i < M; i++) if (inSel[i]) selPos.push_back(i);
    vector<double> crit(selPos.size(), 0.0);
    vector<double> util;
    for (int c = 0; c < t.K; c++) {
        solveCase(t, selIds, selArea, selLen, c, util);
        for (int p = 0; p < (int)selPos.size(); p++) crit[p] = max(crit[p], util[p]);
    }
    vector<int> order(selPos.size());
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b) { return crit[a] > crit[b]; });

    bool changed = true;
    while (changed && remaining > 1e-9) {
        changed = false;
        for (int oi : order) {
            int i = selPos[oi];
            if (areaIdx[i] < t.NAREA - 1) {
                double step = (t.AREA[areaIdx[i] + 1] - t.AREA[areaIdx[i]]) * lenv[i];
                if (remaining + 1e-9 >= step) { remaining -= step; areaIdx[i]++; changed = true; }
            }
        }
    }

    int S = 0;
    for (int i = 0; i < M; i++) if (inSel[i]) S++;
    printf("%d\n", S);
    for (int i = 0; i < M; i++) if (inSel[i]) printf("%d %d\n", ids[i], areaIdx[i]);
    return 0;
}
