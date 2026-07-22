// TIER: strong
// The insight: the checker's objective is F = max_u C(u,t) with
// C(u,t) = 2*m*R(u,t)  (mass times effective resistance -- the exact
// commute-time identity). Adding a shortcut is a RANK-1 update to the
// t-grounded Laplacian's inverse (Sherman-Morrison), so we can afford to
// evaluate, EXACTLY, what every remaining candidate would do to the TRUE
// global worst case before committing to it -- not a hop-count proxy, and
// not a per-vertex local view. We greedily commit to whichever candidate
// most reduces the predicted global max, and we STOP the moment no
// candidate would actually improve it (never burn budget "just because it's
// left"). This is exactly what exposes the ratio game: two symmetric
// bottleneck arms are each individually barely worth their added mass, but
// the single edge CROSSING them halves both resistances for one unit of
// mass and clears the true global worst case in one shot.
#include <bits/stdc++.h>
using namespace std;

int n, m0, t, k, M;
vector<pair<int,int>> cand;
vector<pair<int,int>> origE;

int main(){
    cin >> n >> m0 >> t >> k >> M;
    origE.resize(m0);
    for (int i = 0; i < m0; i++) cin >> origE[i].first >> origE[i].second;
    cand.resize(M);
    for (int i = 0; i < M; i++) cin >> cand[i].first >> cand[i].second;

    int N = n - 1;
    vector<int> ridx(n + 1, -1);
    { int c = 0; for (int v = 1; v <= n; v++) if (v != t) ridx[v] = c++; }

    vector<vector<double>> A(N, vector<double>(N, 0.0));
    auto addEdge = [&](int u, int v, vector<vector<double>>& mat){
        if (u == t && v == t) return;
        if (u == t){ int rv = ridx[v]; mat[rv][rv] += 1.0; return; }
        if (v == t){ int ru = ridx[u]; mat[ru][ru] += 1.0; return; }
        int ru = ridx[u], rv = ridx[v];
        mat[ru][ru] += 1.0; mat[rv][rv] += 1.0; mat[ru][rv] -= 1.0; mat[rv][ru] -= 1.0;
    };
    for (auto &e : origE) addEdge(e.first, e.second, A);

    // Gauss-Jordan inversion -> Minv = A^{-1}
    vector<vector<double>> Minv(N, vector<double>(N, 0.0));
    for (int i = 0; i < N; i++) Minv[i][i] = 1.0;
    {
        vector<vector<double>> W = A;
        for (int col = 0; col < N; col++){
            int piv = col; double best = fabs(W[col][col]);
            for (int r = col + 1; r < N; r++){ double v = fabs(W[r][col]); if (v > best){ best = v; piv = r; } }
            if (piv != col){ swap(W[piv], W[col]); swap(Minv[piv], Minv[col]); }
            double d = W[col][col];
            if (fabs(d) < 1e-12) d = (d < 0 ? -1e-12 : 1e-12);
            for (int j = 0; j < N; j++){ W[col][j] /= d; Minv[col][j] /= d; }
            for (int r = 0; r < N; r++){
                if (r == col) continue;
                double factor = W[r][col];
                if (factor == 0.0) continue;
                for (int j = 0; j < N; j++){ W[r][j] -= factor * W[col][j]; Minv[r][j] -= factor * Minv[col][j]; }
            }
        }
    }

    double mass = (double)m0;
    vector<char> usedCand(M + 1, 0);
    vector<int> chosen;

    auto reducedOf = [&](int v)->int{ return (v == t ? -1 : ridx[v]); };

    for (int step = 0; step < k; step++){
        double curMax = 0.0;
        for (int i = 0; i < N; i++) curMax = max(curMax, Minv[i][i]);
        double curW = 2.0 * mass * curMax;

        int bestIdx = -1;
        double bestW = curW;
        vector<double> bestV;
        double bestDenom = 1.0;

        for (int i = 1; i <= M; i++){
            if (usedCand[i]) continue;
            int a = cand[i - 1].first, b = cand[i - 1].second;
            int ra = reducedOf(a), rb = reducedOf(b);
            if (ra == -1 && rb == -1) continue; // both endpoints == t, degenerate
            vector<double> v(N);
            double s;
            if (ra != -1 && rb != -1){
                for (int j = 0; j < N; j++) v[j] = Minv[j][ra] - Minv[j][rb];
                s = v[ra] - v[rb];
            } else if (ra != -1){ // b == t
                for (int j = 0; j < N; j++) v[j] = Minv[j][ra];
                s = v[ra];
            } else { // a == t
                for (int j = 0; j < N; j++) v[j] = Minv[j][rb];
                s = v[rb];
            }
            double denom = 1.0 + s;
            if (denom < 1e-9) denom = 1e-9;
            double newMax = 0.0;
            for (int j = 0; j < N; j++){
                double nd = Minv[j][j] - v[j] * v[j] / denom;
                if (nd < 0) nd = 0;
                newMax = max(newMax, nd);
            }
            double newW = 2.0 * (mass + 1.0) * newMax;
            if (newW < bestW - 1e-9){
                bestW = newW; bestIdx = i; bestV = v; bestDenom = denom;
            }
        }

        if (bestIdx == -1) break; // nothing left actually helps -> stop (don't dilute for nothing)

        // commit: apply the rank-1 Sherman-Morrison update to the FULL inverse
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                Minv[i][j] -= bestV[i] * bestV[j] / bestDenom;
        mass += 1.0;
        usedCand[bestIdx] = 1;
        chosen.push_back(bestIdx);
    }

    cout << chosen.size() << "\n";
    for (size_t i = 0; i < chosen.size(); i++) cout << chosen[i] << (i + 1 < chosen.size() ? ' ' : '\n');
    if (chosen.empty()) cout << "\n";
    return 0;
}
