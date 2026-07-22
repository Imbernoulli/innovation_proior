#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Quarrelsome Metronomes on a Coupled Table".
//
// Input:  n m ; m edges (u v) ; n metronome deviations f[1..n] (sum f = 0).
// Output: a permutation p[1..n] -- p_i = index of the metronome seated at seat i.
//
// Objective (MIN): let omega_i = f[p_i]. Solve the grounded Laplacian balance
//   equations (x_1 = 0, for every seat i: sum_{j~i} (x_i - x_j) = omega_i) via
//   dense Gaussian elimination on the (n-1)x(n-1) reduced Laplacian (SPD for a
//   connected graph). F = max over edges (u,v) of |x_u - x_v|.
//
// Baseline B (checker-computed): the SAME balance equations solved for the
//   do-nothing seating omega = f (p_i = i), i.e. exactly what solutions/trivial.cpp
//   reproduces -> ratio ~= 0.1.
//
// Score (min): sc = min(1000, 100*B/max(F,eps)); ratio = sc/1000.
// -----------------------------------------------------------------------------

static int n, m;
static vector<int> eu, ev;
static vector<vector<int>> adj;

// Solve the grounded Laplacian system for a given omega (1-indexed, size n+1).
// Returns potentials x (1-indexed, size n+1) with x[1] = 0.
static vector<double> solvePotentials(const vector<ll> &omega) {
    int sz = n - 1; // unknowns x_2..x_n
    vector<vector<double>> A(sz, vector<double>(sz, 0.0));
    vector<double> rhs(sz, 0.0);
    for (int i = 2; i <= n; i++) {
        int ri = i - 2;
        rhs[ri] = (double)omega[i];
        A[ri][ri] += (double)adj[i].size();
        for (int nb : adj[i]) {
            if (nb == 1) continue; // x_1 pinned to 0
            A[ri][nb - 2] -= 1.0;
        }
    }
    // Gaussian elimination with partial pivoting (matrix is SPD -> no singularity
    // for a connected graph; the pivot guard below is defense-in-depth only).
    for (int col = 0; col < sz; col++) {
        int piv = col;
        double best = fabs(A[col][col]);
        for (int r = col + 1; r < sz; r++)
            if (fabs(A[r][col]) > best) { best = fabs(A[r][col]); piv = r; }
        if (piv != col) { swap(A[piv], A[col]); swap(rhs[piv], rhs[col]); }
        double d = A[col][col];
        if (fabs(d) < 1e-12) d = (d >= 0 ? 1e-12 : -1e-12);
        for (int r = col + 1; r < sz; r++) {
            double factor = A[r][col] / d;
            if (factor == 0.0) continue;
            for (int c = col; c < sz; c++) A[r][c] -= factor * A[col][c];
            rhs[r] -= factor * rhs[col];
        }
    }
    vector<double> sol(sz, 0.0);
    for (int r = sz - 1; r >= 0; r--) {
        double s = rhs[r];
        for (int c = r + 1; c < sz; c++) s -= A[r][c] * sol[c];
        double d = A[r][r];
        if (fabs(d) < 1e-12) d = (d >= 0 ? 1e-12 : -1e-12);
        sol[r] = s / d;
    }
    vector<double> x(n + 1, 0.0);
    for (int i = 2; i <= n; i++) x[i] = sol[i - 2];
    return x;
}

static double maxFlow(const vector<double> &x) {
    double F = 0.0;
    for (int i = 0; i < m; i++) {
        double f = fabs(x[eu[i]] - x[ev[i]]);
        if (f > F) F = f;
    }
    return F;
}

int main(int argc, char *argv[]) {
    registerTestlibCmd(argc, argv);

    n = inf.readInt();
    m = inf.readInt();
    eu.resize(m); ev.resize(m);
    adj.assign(n + 1, {});
    for (int i = 0; i < m; i++) {
        eu[i] = inf.readInt();
        ev[i] = inf.readInt();
        adj[eu[i]].push_back(ev[i]);
        adj[ev[i]].push_back(eu[i]);
    }
    vector<ll> freqArr(n + 1);
    for (int i = 1; i <= n; i++) freqArr[i] = inf.readLong();

    // ---- internal baseline B: do-nothing seating omega = freqArr ----
    vector<double> xB = solvePotentials(freqArr);
    double B = maxFlow(xB);
    if (!isfinite(B) || B < 1e-9) B = 1e-9;

    // ---- participant output: permutation p[1..n] (strict bijection) ----
    vector<int> p(n + 1);
    vector<char> seen(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        int v = ouf.readInt(1, n, "p_i");
        if (seen[v]) quitf(_wa, "metronome %d assigned to more than one seat", v);
        seen[v] = 1;
        p[i] = v;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens after permutation");

    vector<ll> omega(n + 1);
    for (int i = 1; i <= n; i++) omega[i] = freqArr[p[i]];

    vector<double> xF = solvePotentials(omega);
    double F = maxFlow(xF);
    if (!isfinite(F)) quitf(_wa, "non-finite objective");

    double sc = min(1000.0, 100.0 * B / max(F, 1e-9));
    quitp(sc / 1000.0, "OK F=%.6f B=%.6f Ratio: %.6f", F, B, sc / 1000.0);
}
