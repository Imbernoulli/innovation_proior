#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// -----------------------------------------------------------------------------
// Checker / scorer for "Day-Night Courier Laplacians".
//
// Input:
//   n m
//   tau
//   capDay capNight
//   m lines: u v w   (1-indexed edge, weight w)
//
// Output: m integers (0=day shift "A", 1=night shift "B"), one per edge in input order.
// Feasibility: #day-edges <= capDay, #night-edges <= capNight (every edge assigned exactly once
//   because the participant emits exactly one token per input edge).
//
// Objective (MIN): let L_A, L_B be the (weighted) graph Laplacians of the day-only and
//   night-only subgraphs on the n hubs. One day+night period maps a disagreement vector x
//   (x with mean 0) to  M x = exp(-tau*L_B) exp(-tau*L_A) x.  Because L_A, L_B generally do NOT
//   commute, order matters. F = the Floquet multiplier = the spectral radius of M restricted to
//   the disagreement subspace (orthogonal complement of the all-ones vector) -- the per-period
//   worst-case contraction factor. All eigenvalues of M are real and positive (M is similar to
//   exp(-tau L_A/2) exp(-tau L_B) exp(-tau L_A/2), an SPD matrix), so a deterministic power
//   iteration with mean-deflation converges to F monotonically-ish and safely.
//
//   exp(-tau*L) is applied to vectors matrix-free via scaling + a truncated Taylor series
//   (no explicit n x n matrix is ever formed): pick s with delta = tau/2^s small enough that
//   delta*lambdaBound <= SUBSTEP_BOUND, apply the K-term Taylor series for exp(-delta L) to the
//   vector 2^s times in a row (mathematically exact as 2^s, K -> infinity; K=12 and
//   SUBSTEP_BOUND=0.5 give double-precision accuracy for any tau, lambdaBound we generate).
//
// Baseline B: same F, computed for the checker's own canonical "trivial" partition (first
//   capDay edges in input order go to day, the rest to night). The trivial reference solution
//   reproduces exactly this construction, so it scores ~0.1.
//
// Score (min-objective convention): sc = min(1000, 100*B/F); Ratio = sc/1000, capped at 1.0.
// -----------------------------------------------------------------------------

struct Edge { int u, v; double w; };

static int N;
static const int KTERMS = 12;
static const double SUBSTEP_BOUND = 0.5;
static const int POWER_ITERS = 60;

// y = L * x   (x,y size N; L is the Laplacian implied by `edges`)
static void applyL(const vector<Edge>& edges, const vector<double>& x, vector<double>& y) {
    fill(y.begin(), y.end(), 0.0);
    for (const auto& e : edges) {
        double diff = (x[e.u] - x[e.v]) * e.w;
        y[e.u] += diff;
        y[e.v] -= diff;
    }
}

// returns exp(-tau*L) applied to x, matrix-free, via scaling + K-term Taylor.
static vector<double> applyExpL(const vector<Edge>& edges, double tau, double lambdaBound,
                                 vector<double> x) {
    if (tau <= 0.0 || edges.empty()) return x;
    int s = 0;
    double delta = tau;
    while (delta * lambdaBound > SUBSTEP_BOUND && s < 40) { delta *= 0.5; s++; }
    long long reps = 1LL << s;
    vector<double> cur = x, term(N), Lterm(N), y(N);
    for (long long r = 0; r < reps; r++) {
        term = cur;
        y = cur;
        for (int j = 1; j <= KTERMS; j++) {
            applyL(edges, term, Lterm);
            double coef = -delta / j;
            for (int i = 0; i < N; i++) term[i] = coef * Lterm[i];
            for (int i = 0; i < N; i++) y[i] += term[i];
        }
        cur = y;
    }
    return cur;
}

// deterministic power iteration for the largest eigenvalue of M = exp(-tau LB) exp(-tau LA)
// restricted to the orthogonal complement of the all-ones vector, from a GIVEN start vector.
static double floquetRadiusFrom(vector<double> v, const vector<Edge>& dayE,
                                 const vector<Edge>& nightE, double tau, double lambdaBound) {
    double mean = 0;
    for (int i = 0; i < N; i++) mean += v[i];
    mean /= N;
    for (int i = 0; i < N; i++) v[i] -= mean;
    double nrm0 = 0;
    for (int i = 0; i < N; i++) nrm0 += v[i] * v[i];
    nrm0 = sqrt(nrm0);
    if (nrm0 < 1e-12) return 0.0;
    for (int i = 0; i < N; i++) v[i] /= nrm0;

    double rho = 0.0;
    for (int it = 0; it < POWER_ITERS; it++) {
        v = applyExpL(dayE, tau, lambdaBound, v);
        v = applyExpL(nightE, tau, lambdaBound, v);
        double m2 = 0;
        for (int i = 0; i < N; i++) m2 += v[i];
        m2 /= N;
        for (int i = 0; i < N; i++) v[i] -= m2;
        double nrm = 0;
        for (int i = 0; i < N; i++) nrm += v[i] * v[i];
        nrm = sqrt(nrm);
        if (nrm < 1e-300) { rho = 0.0; break; }
        rho = nrm;
        for (int i = 0; i < N; i++) v[i] = v[i] / nrm;
    }
    return rho;
}

// A single power-iteration trajectory from a smooth start vector can have near-zero overlap
// with a LOCALIZED slow eigenvector (e.g. one hidden behind a fragmented/adversarial partition),
// underestimating the true spectral radius. Guard against that by racing several structurally
// different deterministic start vectors and reporting the largest estimate (a valid lower bound
// on the true radius from each start, so the max is still a safe -- and far more robust --
// estimate of F).
static double floquetRadius(const vector<Edge>& dayE, const vector<Edge>& nightE, double tau,
                             double lambdaBound) {
    double best = 0.0;
    // start 1: smooth ramp (catches global/low-frequency slow modes)
    {
        vector<double> v(N);
        for (int i = 0; i < N; i++) v[i] = (double)i - (N - 1) / 2.0;
        best = max(best, floquetRadiusFrom(v, dayE, nightE, tau, lambdaBound));
    }
    // start 2: alternating +-1 (catches high-frequency / bipartite-like slow modes)
    {
        vector<double> v(N);
        for (int i = 0; i < N; i++) v[i] = (i % 2 == 0) ? 1.0 : -1.0;
        best = max(best, floquetRadiusFrom(v, dayE, nightE, tau, lambdaBound));
    }
    // start 3: fixed deterministic pseudo-random vector (generic overlap with ANY localized
    // mode, including ones hidden from starts 1 and 2 by an adversarial/fragmented partition)
    {
        vector<double> v(N);
        unsigned long long x = 88172645463325252ULL;
        for (int i = 0; i < N; i++) {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            v[i] = (double)((x % 2000001ULL)) / 1000000.0 - 1.0;
        }
        best = max(best, floquetRadiusFrom(v, dayE, nightE, tau, lambdaBound));
    }
    return best;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();
    double tau = inf.readDouble();
    long long capDay = inf.readLong();
    long long capNight = inf.readLong();
    N = n;

    vector<Edge> ev(m);
    double maxDeg = 0;
    vector<double> deg(n, 0.0);
    for (int i = 0; i < m; i++) {
        int u = inf.readInt(1, n, "u") - 1;
        int v = inf.readInt(1, n, "v") - 1;
        if (u == v) quitf(_fail, "generator bug: self loop");
        double w = inf.readInt(1, 1000000000, "w");
        ev[i] = {u, v, w};
        deg[u] += w; deg[v] += w;
    }
    for (int i = 0; i < n; i++) maxDeg = max(maxDeg, deg[i]);
    double lambdaBound = max(2.0 * maxDeg, 1e-9);

    // ---- read & validate participant output ----
    vector<int> shift(m);
    long long cntDay = 0, cntNight = 0;
    for (int i = 0; i < m; i++) {
        int s = ouf.readInt(0, 1, "shift");
        shift[i] = s;
        if (s == 0) cntDay++; else cntNight++;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after %d shift assignments", m);
    if (cntDay > capDay) quitf(_wa, "day shift has %lld edges > cap %lld", cntDay, capDay);
    if (cntNight > capNight) quitf(_wa, "night shift has %lld edges > cap %lld", cntNight, capNight);

    vector<Edge> dayE, nightE;
    dayE.reserve(m); nightE.reserve(m);
    for (int i = 0; i < m; i++) {
        if (shift[i] == 0) dayE.push_back(ev[i]); else nightE.push_back(ev[i]);
    }

    double F = floquetRadius(dayE, nightE, tau, lambdaBound);
    if (!isfinite(F)) quitf(_wa, "non-finite objective (bad output shape)");
    if (F < 1e-9) F = 1e-9;

    // ---- checker's own trivial baseline: first capDay edges (input order) -> day ----
    vector<Edge> dayR, nightR;
    dayR.reserve(m); nightR.reserve(m);
    for (int i = 0; i < m; i++) {
        if ((long long)i < capDay) dayR.push_back(ev[i]); else nightR.push_back(ev[i]);
    }
    double B = floquetRadius(dayR, nightR, tau, lambdaBound);
    if (B < 1e-9) B = 1e-9;

    double sc = min(1000.0, 100.0 * B / F);
    quitp(sc / 1000.0, "OK F=%.6f B=%.6f cntDay=%lld cntNight=%lld Ratio: %.6f",
          F, B, cntDay, cntNight, sc / 1000.0);
    return 0;
}
