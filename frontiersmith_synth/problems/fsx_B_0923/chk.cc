#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// -----------------------------------------------------------------------------
// Checker / scorer for "Charging Depots for the Bridge City"
//   family: commute-time-depots
//
// Input:  n m k ; then n population weights p_1..p_n ; then m edges "u v"
//         (1-indexed, undirected, simple, connected).
// Output: k distinct vertex ids in [1,n] -- the chosen depot set S.
//
// Objective (MIN): for every hub v define its access time a(v) as the unique
// solution of the classic hitting-time recurrence for a simple symmetric
// random walk absorbed the instant it reaches ANY depot:
//     a(v) = 0                                            if v in S
//     a(v) = 1 + (1/deg(v)) * sum_{u ~ v} a(u)             if v not in S
// Equivalently, on non-depot rows this is the linear system
//     deg(v) * a(v)  -  sum_{u ~ v, u not in S} a(u)  =  deg(v)
// a principal submatrix of the graph Laplacian restricted to V\S (SPD since
// the graph is connected and S is non-empty) with right-hand side deg(v).
// We solve it exactly via dense Gaussian elimination with partial pivoting
// (|V\S| <= n <= ~450, so O(n^3) is comfortably fast and numerically stable
// for these mildly-conditioned instances).
//
// F = population-weighted average access time = sum_v p_v*a(v) / sum_v p_v.
// Minimize F.
//
// Internal baseline B (checker-computed "do nothing smart", MIN-convention):
// the BETTER (lower-F) of two fixed, resistance-oblivious constructions: (a)
// k vertex ids evenly spaced by RAW ID number, and (b) the k highest-
// population hubs (ties by smaller id). Taking the better of the two keeps B
// from being a fluke-weak reference (population top-k can accidentally
// cluster; id-spacing can land mid-corridor) while still being completely
// blind to the road network's resistance/access geometry -- neither
// construction ever deliberately reaches out along a bridge. trivial.cpp is
// deliberately worse than both -> ratio ~ 0.1.
//
// Score: sc = min(1000, 100*B/max(eps,F)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int n, m, k;
vector<int> pop_;
vector<vector<int>> adj;
vector<int> deg_;

// Solve for access times given depot set S (as a boolean membership array),
// return the population-weighted average access time.
double objectiveOf(const vector<char>& isDepot){
    vector<int> ridx(n + 1, -1), orig;
    orig.reserve(n);
    for (int v = 1; v <= n; v++) if (!isDepot[v]){ ridx[v] = (int)orig.size(); orig.push_back(v); }
    int N = (int)orig.size();
    if (N == 0) return 0.0; // all vertices are depots (only possible if k==n)

    vector<vector<double>> A(N, vector<double>(N, 0.0));
    vector<double> b(N, 0.0);
    for (int idx = 0; idx < N; idx++){
        int v = orig[idx];
        A[idx][idx] += (double)deg_[v];
        b[idx] = (double)deg_[v];
        for (int u : adj[v]) if (!isDepot[u]) A[idx][ridx[u]] -= 1.0;
    }

    // Gaussian elimination with partial pivoting, single RHS.
    for (int col = 0; col < N; col++){
        int piv = col; double best = fabs(A[col][col]);
        for (int r = col + 1; r < N; r++){
            double v = fabs(A[r][col]);
            if (v > best){ best = v; piv = r; }
        }
        if (piv != col){ swap(A[piv], A[col]); swap(b[piv], b[col]); }
        double d = A[col][col];
        if (fabs(d) < 1e-12) d = (d < 0 ? -1e-12 : 1e-12);
        for (int j = col; j < N; j++) A[col][j] /= d;
        b[col] /= d;
        for (int r = 0; r < N; r++){
            if (r == col) continue;
            double factor = A[r][col];
            if (factor == 0.0) continue;
            for (int j = col; j < N; j++) A[r][j] -= factor * A[col][j];
            b[r] -= factor * b[col];
        }
    }

    vector<double> a(n + 1, 0.0);
    for (int idx = 0; idx < N; idx++) a[orig[idx]] = b[idx];

    double num = 0.0, den = 0.0;
    for (int v = 1; v <= n; v++){ num += (double)pop_[v] * a[v]; den += (double)pop_[v]; }
    return num / max(1e-9, den);
}

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    n = inf.readInt(); m = inf.readInt(); k = inf.readInt();
    pop_.assign(n + 1, 0);
    for (int v = 1; v <= n; v++) pop_[v] = inf.readInt(1, 1000000, "p_v");
    adj.assign(n + 1, {});
    deg_.assign(n + 1, 0);
    for (int i = 0; i < m; i++){
        int u = inf.readInt(1, n, "u");
        int v = inf.readInt(1, n, "v");
        adj[u].push_back(v); adj[v].push_back(u);
        deg_[u]++; deg_[v]++;
    }

    vector<char> isDepot(n + 1, 0);
    vector<char> used(n + 1, 0);
    for (int i = 0; i < k; i++){
        int idx = ouf.readInt(1, n, "depot");
        if (used[idx]) quitf(_wa, "depot vertex %d chosen more than once", idx);
        used[idx] = 1;
        isDepot[idx] = 1;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens in output");

    double F = objectiveOf(isDepot);
    if (!isfinite(F) || F < 0.0) quitf(_wa, "non-finite or negative objective");

    vector<char> baseIdSpread(n + 1, 0);
    {
        vector<char> usedB(n + 1, 0);
        for (int i = 0; i < k; i++){
            int idx = 1 + (int)((long long)i * n / k);
            if (idx < 1) idx = 1; if (idx > n) idx = n;
            while (usedB[idx]) idx = (idx % n) + 1; // extremely defensive; shouldn't trigger
            usedB[idx] = 1; baseIdSpread[idx] = 1;
        }
    }
    vector<char> basePopTop(n + 1, 0);
    {
        vector<int> order(n);
        for (int v = 1; v <= n; v++) order[v - 1] = v;
        sort(order.begin(), order.end(), [&](int a, int b){
            if (pop_[a] != pop_[b]) return pop_[a] > pop_[b];
            return a < b;
        });
        for (int i = 0; i < k; i++) basePopTop[order[i]] = 1;
    }
    double B = min(objectiveOf(baseIdSpread), objectiveOf(basePopTop));
    if (!isfinite(B) || B <= 0.0) quitf(_wa, "internal error: non-positive baseline");

    double sc = min(1000.0, 100.0 * B / max(1e-9, F));
    quitp(sc / 1000.0, "OK F=%.6f B=%.6f Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
