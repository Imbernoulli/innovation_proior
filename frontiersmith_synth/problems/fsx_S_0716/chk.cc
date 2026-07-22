#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// chk.cc -- scorer for Wafer Stepper Exposure Order (minimax self-generated heat).
// Validates: participant output is a permutation of 1..N.  Objective:
//   temp(t) = sum_{k<t} Q0*w[p_k]*exp(-alpha*(t-k))*exp(-r2(p_t,p_k)/(4*D*(t-k))) / (4*pi*D*(t-k))
//   F = max_t temp(t)                                          (minimization)
// Baseline B = F of the INPUT-ORDER schedule p_k = k (always feasible, positive).
// ratio = min(1, (B / max(1e-9,F)) / 10).

static int N;
static vector<double> X, Y;
static vector<double> W;
static double D, alpha, Q0;

static double simulate(const vector<int>& p) {
    // p[t] (0-indexed t=0..N-1) is the 0-indexed field exposed at step t+1.
    double worst = 0.0;
    for (int t = 1; t < N; t++) {
        int a = p[t];
        double temp = 0.0;
        for (int k = 0; k < t; k++) {
            int b = p[k];
            double dt = (double)(t - k);
            double dx = X[a] - X[b], dy = Y[a] - Y[b];
            double r2 = dx * dx + dy * dy;
            double denom = 4.0 * D * dt;
            double contrib = Q0 * W[b] * exp(-alpha * dt) * exp(-r2 / denom) / (4.0 * M_PI * D * dt);
            temp += contrib;
        }
        if (temp > worst) worst = temp;
    }
    return worst;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    N = inf.readInt();
    X.assign(N, 0.0);
    Y.assign(N, 0.0);
    W.assign(N, 0.0);
    D = inf.readDouble();
    alpha = inf.readDouble();
    Q0 = inf.readDouble();
    for (int i = 0; i < N; i++) {
        X[i] = (double)inf.readInt();
        Y[i] = (double)inf.readInt();
        W[i] = (double)inf.readInt();
    }

    // ---- read & validate participant permutation ----
    vector<int> p(N);
    vector<char> seen(N + 1, 0);
    for (int t = 0; t < N; t++) {
        int v = ouf.readInt(1, N, "p_i");
        if (seen[v]) quitf(_wa, "value %d repeated in the exposure order", v);
        seen[v] = 1;
        p[t] = v - 1;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after the exposure order");

    double F = simulate(p);
    if (!isfinite(F)) quitf(_wa, "non-finite objective");

    // ---- baseline B: input-order schedule p_k = k ----
    vector<int> ident(N);
    for (int i = 0; i < N; i++) ident[i] = i;
    double B = simulate(ident);
    if (!(B > 0.0) || !isfinite(B)) B = 1e-9;

    double sc = min(1000.0, 100.0 * B / max(1e-9, F));
    quitp(sc / 1000.0, "OK F=%.6f B=%.6f Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
