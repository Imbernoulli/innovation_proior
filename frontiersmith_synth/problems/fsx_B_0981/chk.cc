#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// -----------------------------------------------------------------------------
// Checker / scorer for "Bessel Sideband Inversion" (fm-synth-timbre-matching).
//
// Input:  K H rc lambda ; then K lines "Rlo_i Rhi_i Cmax_i" ; then H reals T(1..H).
// Output (participant): K lines "r_i I_i" (integer ratio in window, real index in cap).
//
// Reference spectrum: one carrier (ratio rc) phase-modulated in PARALLEL by K
// modulators (ratio r_i, index I_i). Jacobi-Anger expansion places energy at every
// harmonic h reachable by a signed-order combination (n_1..n_K), |n_i|<=NMAX, with
// rc + sum n_i*r_i = h; the coefficient of that combination is prod_i J_{n_i}(I_i).
// Combinations landing on the same h add (with sign) -> real constructive/destructive
// interference. A(h) = |sum of those coefficients|, for h=1..H.
//
// Objective (MIN): F = sum_h (A(h)-T(h))^2 + lambda * sum_i I_i.
// Baseline B (checker-computed do-nothing): all r_i at window minimum, all I_i=0
// (silent modulation, only the bare carrier sounds) -> F of that construction.
// Score (min): sc = min(1000, 100*B/max(eps,F)); ratio = sc/1000.
// -----------------------------------------------------------------------------

static const int NMAX = 10;

// Precompute J_n(x) for n=-NMAX..NMAX into out[NMAX+n].
static void besselRow(double x, vector<double> &out) {
    out.assign(2 * NMAX + 1, 0.0);
    for (int n = 0; n <= NMAX; n++) {
        double v = std::cyl_bessel_j((double)n, x);
        out[NMAX + n] = v;
        out[NMAX - n] = (n % 2 == 0) ? v : -v;   // J_{-n}(x) = (-1)^n J_n(x)
    }
}

// Recursive enumeration over K operators' sideband orders, accumulating signed
// contributions into bin[0..H] (bin[h] corresponds to harmonic h, folded via abs).
static void enumerate(int pos, int K, long long hsum, double prod,
                       const vector<vector<double>> &J, const vector<int> &ratio,
                       int H, vector<double> &bin) {
    if (pos == K) {
        long long ah = llabs(hsum);
        if (ah <= H) bin[(int)ah] += prod;
        return;
    }
    for (int n = -NMAX; n <= NMAX; n++) {
        double c = J[pos][NMAX + n];
        if (c == 0.0) continue;
        enumerate(pos + 1, K, hsum + (long long)n * ratio[pos], prod * c, J, ratio, H, bin);
    }
}

static vector<double> synthesize(int K, long long rc, int H, const vector<int> &ratio,
                                  const vector<double> &idx) {
    vector<vector<double>> J(K);
    for (int i = 0; i < K; i++) besselRow(idx[i], J[i]);
    vector<double> bin(H + 1, 0.0);
    enumerate(0, K, rc, 1.0, J, ratio, H, bin);
    vector<double> A(H + 1, 0.0);
    for (int h = 1; h <= H; h++) A[h] = fabs(bin[h]);
    return A;
}

int main(int argc, char *argv[]) {
    registerTestlibCmd(argc, argv);

    int K = inf.readInt();
    int H = inf.readInt();
    int rc = inf.readInt();
    double lambda = inf.readDouble();

    vector<int> Rlo(K), Rhi(K);
    vector<double> Cmax(K);
    for (int i = 0; i < K; i++) {
        Rlo[i] = inf.readInt();
        Rhi[i] = inf.readInt();
        Cmax[i] = inf.readDouble();
    }
    vector<double> T(H + 1, 0.0);
    for (int h = 1; h <= H; h++) T[h] = inf.readDouble();

    // ---- internal baseline B: all ratios at window minimum, all indices 0 ----
    {
        vector<int> ratioB(Rlo);
        vector<double> idxB(K, 0.0);
        vector<double> Ab = synthesize(K, rc, H, ratioB, idxB);
        double Fb = 0.0;
        for (int h = 1; h <= H; h++) { double d = Ab[h] - T[h]; Fb += d * d; }
        double B = Fb;   // cost term is 0 for the baseline

        // ---- read participant output: strict feasibility ----
        vector<int> ratio(K);
        vector<double> idx(K);
        for (int i = 0; i < K; i++) {
            ratio[i] = ouf.readInt(Rlo[i], Rhi[i], format("r%d", i + 1).c_str());
            double v = ouf.readDouble();
            if (!isfinite(v)) quitf(_wa, "index %d is not finite", i + 1);
            if (v < -1e-9 || v > Cmax[i] + 1e-9)
                quitf(_wa, "index %d = %.6f violates cap [0,%.6f]", i + 1, v, Cmax[i]);
            idx[i] = max(0.0, min(v, Cmax[i]));
        }
        if (!ouf.seekEof()) quitf(_wa, "trailing output after operator list");

        vector<double> A = synthesize(K, rc, H, ratio, idx);
        double F = 0.0;
        for (int h = 1; h <= H; h++) { double d = A[h] - T[h]; F += d * d; }
        double cost = 0.0;
        for (int i = 0; i < K; i++) cost += idx[i];
        F += lambda * cost;

        if (B < 1e-9) B = 1e-9;
        double sc = min(1000.0, 100.0 * B / max(1e-9, F));
        quitp(sc / 1000.0, "OK F=%.6f B=%.6f Ratio: %.6f", F, B, sc / 1000.0);
    }
    return 0;
}
