// TIER: strong
// The insight: with resistances FIXED, minimizing sum r_e*f_e^2 under fixed
// nodal injections is exactly Thomson's principle -- the minimizer is the
// electrical-current equilibrium of the resistor network, found by solving one
// Laplacian linear system (no search over splits needed). Booster siting then
// perturbs a handful of resistances; instead of ranking candidate pipes by their
// PRE-boost looks, re-solve the equilibrium for every tentative boost and keep
// whichever one lowers the RE-EQUILIBRATED total dissipation the most. Repeat
// until K boosters are placed, then emit the final equilibrium flows.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int V, E, S, T, K;
vector<int> eu, ev;
vector<ll> er, egain;
vector<int> ecand;
vector<ll> srcAmt, sinkAmt; // indexed by node, -1 if not that role... actually use maps

vector<double> rhs;

// Solve the weighted Laplacian for nodal potentials given per-edge resistance.
// Fixes p[1] = 0 as the reference potential.
vector<double> solvePotentials(const vector<double>& res) {
    vector<vector<double>> L(V + 1, vector<double>(V + 1, 0.0));
    for (int e = 0; e < E; e++) {
        double g = 1.0 / res[e];
        L[eu[e]][eu[e]] += g; L[ev[e]][ev[e]] += g;
        L[eu[e]][ev[e]] -= g; L[ev[e]][eu[e]] -= g;
    }
    int ref = 1;
    vector<int> idx;
    for (int v = 1; v <= V; v++) if (v != ref) idx.push_back(v);
    int m = (int)idx.size();
    vector<vector<double>> A(m, vector<double>(m + 1, 0.0));
    for (int i = 0; i < m; i++) {
        int vi = idx[i];
        for (int j = 0; j < m; j++) A[i][j] = L[vi][idx[j]];
        A[i][m] = rhs[vi];
    }
    // Gaussian elimination with partial pivoting
    for (int c = 0; c < m; c++) {
        int piv = c;
        for (int r = c + 1; r < m; r++) if (fabs(A[r][c]) > fabs(A[piv][c])) piv = r;
        swap(A[c], A[piv]);
        double d = A[c][c];
        if (fabs(d) < 1e-12) continue;
        for (int r = 0; r < m; r++) {
            if (r == c) continue;
            double factor = A[r][c] / d;
            if (factor == 0.0) continue;
            for (int j = c; j <= m; j++) A[r][j] -= factor * A[c][j];
        }
    }
    vector<double> p(V + 1, 0.0);
    for (int i = 0; i < m; i++) {
        double d = A[i][i];
        p[idx[i]] = (fabs(d) < 1e-12) ? 0.0 : A[i][m] / d;
    }
    p[ref] = 0.0;
    return p;
}

double dissipationOf(const vector<double>& res, const vector<double>& p, vector<double>* flowOut = nullptr) {
    double F = 0.0;
    if (flowOut) flowOut->assign(E, 0.0);
    for (int e = 0; e < E; e++) {
        double fl = (p[eu[e]] - p[ev[e]]) / res[e];
        F += res[e] * fl * fl;
        if (flowOut) (*flowOut)[e] = fl;
    }
    return F;
}

int main() {
    scanf("%d %d %d %d %d", &V, &E, &S, &T, &K);
    vector<pair<int,ll>> srcs(S), sinks(T);
    for (int i = 0; i < S; i++) scanf("%d %lld", &srcs[i].first, &srcs[i].second);
    for (int i = 0; i < T; i++) scanf("%d %lld", &sinks[i].first, &sinks[i].second);

    eu.assign(E, 0); ev.assign(E, 0); er.assign(E, 0); ecand.assign(E, 0); egain.assign(E, 0);
    for (int e = 0; e < E; e++)
        scanf("%d %d %lld %d %lld", &eu[e], &ev[e], &er[e], &ecand[e], &egain[e]);

    rhs.assign(V + 1, 0.0);
    for (auto &p : srcs) rhs[p.first] += (double)p.second;
    for (auto &p : sinks) rhs[p.first] -= (double)p.second;

    vector<double> resCur(E);
    for (int e = 0; e < E; e++) resCur[e] = (double)er[e];

    vector<char> boosted(E, 0);
    vector<int> boostedList;
    vector<int> cands;
    for (int e = 0; e < E; e++) if (ecand[e]) cands.push_back(e);

    for (int round = 0; round < K && (int)boostedList.size() < (int)cands.size(); round++) {
        int bestEdge = -1;
        double bestF = 1e300;
        for (int c : cands) {
            if (boosted[c]) continue;
            double save = resCur[c];
            resCur[c] = max(1.0, (double)(er[c] - egain[c]));
            vector<double> p = solvePotentials(resCur);
            double F = dissipationOf(resCur, p);
            if (F < bestF) { bestF = F; bestEdge = c; }
            resCur[c] = save;
        }
        if (bestEdge < 0) break;
        resCur[bestEdge] = max(1.0, (double)(er[bestEdge] - egain[bestEdge]));
        boosted[bestEdge] = 1;
        boostedList.push_back(bestEdge);
    }

    vector<double> pFinal = solvePotentials(resCur);
    vector<double> flow;
    dissipationOf(resCur, pFinal, &flow);

    printf("%d", (int)boostedList.size());
    for (int b : boostedList) printf(" %d", b + 1);
    for (int e = 0; e < E; e++) printf(" %.6f", flow[e]);
    printf("\n");
    return 0;
}
