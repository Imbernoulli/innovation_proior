#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
static const double PI = 3.14159265358979323846;

// -----------------------------------------------------------------------------
// Checker for "Phase-Only Beam Shaping on a Linear Array".
//
// Input:  N M D1000 / a_1..a_N / ang_1..ang_M / tgt_1..tgt_M / w_1..w_M / K /
//         idx_1..idx_K / thresh10000   (see statement.txt / gen.cpp).
// Output: N integers phi_1..phi_N in [0,3599] (0.1-degree units).
//
// Objective (MIN):
//   P_m  = |AF(theta_m)|^2 / Pmax^2   (realized normalized power, phase-only excitation)
//   err  = sum_m w_m*(P_m-T_m)^2 / sum_m w_m
//   pen  = 4/max(1,K) * sum_{m in nulls} max(0,P_m-thresh)^2
//   F    = err + pen        (scaled to an integer via *1e6 for exact ratio math)
//
// Baseline B: same F for the "do-nothing" phi_i=0 construction.
// Score (min): sc = min(1000, 100*B/max(1,F)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int M = inf.readInt();
    int D1000 = inf.readInt();
    double d = D1000 / 1000.0;

    vector<double> a(N);
    for (int i = 0; i < N; i++) a[i] = inf.readInt() / 1000.0;
    double Pmax = 0; for (double v : a) Pmax += v;
    if (Pmax <= 0) Pmax = 1;

    vector<double> thetaRad(M);
    for (int m = 0; m < M; m++) thetaRad[m] = inf.readInt() * PI / 180000.0;

    vector<double> T(M);
    for (int m = 0; m < M; m++) T[m] = inf.readInt() / 10000.0;

    vector<double> W(M);
    double sumW = 0;
    for (int m = 0; m < M; m++){ W[m] = inf.readInt(); sumW += W[m]; }
    if (sumW <= 0) sumW = 1;

    int K = inf.readInt();
    vector<int> nullIdx(K);
    for (int i = 0; i < K; i++) nullIdx[i] = inf.readInt() - 1;   // 0-based

    double thresh = inf.readInt() / 10000.0;

    auto computeF = [&](const vector<int>& phi3600) -> double {
        double err = 0, pen = 0;
        vector<double> P(M);
        for (int m = 0; m < M; m++){
            double re = 0, im = 0;
            for (int i = 0; i < N; i++){
                double ang = phi3600[i] * PI / 1800.0 + 2.0 * PI * d * i * cos(thetaRad[m]);
                re += a[i] * cos(ang);
                im += a[i] * sin(ang);
            }
            P[m] = (re * re + im * im) / (Pmax * Pmax);
            err += W[m] * (P[m] - T[m]) * (P[m] - T[m]);
        }
        err /= sumW;
        for (int idx : nullIdx){
            double over = P[idx] - thresh;
            if (over > 0) pen += over * over;
        }
        pen *= 4.0 / max(1, K);
        return err + pen;
    };

    // ---- read participant output, validate feasibility ----
    vector<int> phi(N);
    for (int i = 0; i < N; i++) phi[i] = ouf.readInt(0, 3599, "phi_i");
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    double Fd = computeF(phi);
    if (!isfinite(Fd)) quitf(_wa, "non-finite objective");
    ll F = (ll)llround(Fd * 1e6);
    if (F < 0) F = 0;

    vector<int> phiZero(N, 0);
    double Bd = computeF(phiZero);
    ll B = (ll)llround(Bd * 1e6);
    if (B <= 0) B = 1;

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
