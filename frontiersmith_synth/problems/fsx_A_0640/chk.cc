#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Wire-bending robot that keeps hitting itself".
//
// Input:  m L c TOL K ; then m lines (station index order i=1..m):
//         x_i theta_i delta_i w_i
//
// Output: m lines "i a_i" -- a permutation of 1..m (the bending ORDER) paired
//         with the APPLIED angle a_i chosen for station i (must satisfy
//         |a_i - theta_i| <= c).
//
// Simulation (replayed identically for the participant AND the internal
// baseline): process stations in the given order. When station i is bent,
// realized_i = a_i + delta_i (springback). reach_i = arc-length distance from
// x_i to the NEAREST station j>i already bent earlier in THIS order (or to the
// free tip L if none yet). chord_i = 2*reach_i*sin(|realized_i| degrees / 2).
// If chord_i > K the bend COLLIDES: station i contributes 0 and stays unshaped
// (tail geometry beyond it is unaffected). Otherwise station i becomes "bent"
// (braces the reach for later bends on its base side) and contributes
// w_i * max(0, 1 - |realized_i - theta_i| / TOL) to the objective F.
//
// Baseline B: the SAME simulation using the natural geometric order 1..m with
// a_i = theta_i (no precompensation) -- "just bend it in part order, ignore
// springback". Score (max): sc = min(1000, 100*F/max(1,B)); ratio = sc/1000.
// -----------------------------------------------------------------------------

static int m;
static ll L;
static int c, TOL;
static ll K;
static vector<ll> X, THETA, DELTA, W; // 1..m ; X[m+1] = L (tip sentinel)

static double simulate(const vector<int>& order, const vector<ll>& a) {
    vector<char> bent(m + 2, 0);
    double F = 0.0;
    for (int idx = 0; idx < (int)order.size(); idx++) {
        int i = order[idx];
        ll realized = a[i] + DELTA[i];
        // nearest already-bent station strictly after i, else the tip (m+1)
        int nxt = m + 1;
        for (int j = i + 1; j <= m + 1; j++) {
            if (j == m + 1 || bent[j]) { nxt = j; break; }
        }
        ll reach = X[nxt] - X[i];
        double chord = 2.0 * (double)reach * sin(fabs((double)realized) * M_PI / 360.0);
        if (chord > (double)K) {
            continue; // collision: contributes 0, stays unshaped
        }
        bent[i] = 1;
        double err = fabs((double)(realized - THETA[i]));
        double q = 1.0 - err / (double)TOL;
        if (q < 0.0) q = 0.0;
        F += (double)W[i] * q;
    }
    return F;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    m = inf.readInt();
    L = inf.readLong();
    c = inf.readInt();
    TOL = inf.readInt();
    K = inf.readLong();
    X.assign(m + 2, 0);
    THETA.assign(m + 1, 0);
    DELTA.assign(m + 1, 0);
    W.assign(m + 1, 0);
    for (int i = 1; i <= m; i++) {
        X[i] = inf.readLong();
        THETA[i] = inf.readLong();
        DELTA[i] = inf.readLong();
        W[i] = inf.readLong();
    }
    X[m + 1] = L;

    // ---- internal baseline B: natural order, zero precompensation ----
    vector<int> baseOrder(m);
    for (int i = 0; i < m; i++) baseOrder[i] = i + 1;
    vector<ll> baseA(m + 1);
    for (int i = 1; i <= m; i++) baseA[i] = THETA[i];
    double B = simulate(baseOrder, baseA);
    if (B <= 0.0) B = 1.0; // guard only; generator keeps a positive baseline

    // ---- read participant's order + applied angles ----
    vector<int> order(m);
    vector<ll> a(m + 1, 0);
    vector<char> seen(m + 1, 0);
    for (int k = 0; k < m; k++) {
        int idx = ouf.readInt(1, m, "station index");
        if (seen[idx]) quitf(_wa, "station %d bent more than once", idx);
        seen[idx] = 1;
        ll ai = ouf.readLong(-100000LL, 100000LL, "applied angle");
        if (llabs(ai - THETA[idx]) > c)
            quitf(_wa, "station %d applied angle %lld outside [theta-c, theta+c] = [%lld,%lld]",
                  idx, ai, THETA[idx] - c, THETA[idx] + c);
        order[k] = idx;
        a[idx] = ai;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    double F = simulate(order, a);

    double sc = min(1000.0, 100.0 * F / max(1.0, B));
    quitp(sc / 1000.0, "OK F=%.4f B=%.4f Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
