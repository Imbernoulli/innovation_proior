#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
static const double PI = 3.14159265358979323846;

// -----------------------------------------------------------------------------
// Generator for "Phase-Only Beam Shaping on a Linear Array"
// family: antenna-array-beam-shaping
//
// Emits: N M D1000 / a_1..a_N / ang_1..ang_M / tgt_1..tgt_M / w_1..w_M / K /
//        idx_1..idx_K / thresh10000
//
// Ladder (testId 1..10):
//   1-2  small, SINGLE off-broadside lobe, few/no nulls        (sanity)
//   3,4,5,7,9,10  multi-lobe targets (2-4 lobes) +/- tapered magnitudes
//                 -> TRAP: a single steered beam cannot reproduce them
//   6,8  PLANTED: target = the actual realizable power pattern of a hidden
//                 random phase-only excitation (with the SAME a_i), so an
//                 alternating-projection solver can recover it almost exactly
//                 while single-beam steering cannot.
// -----------------------------------------------------------------------------

static double AFpow(int N, const vector<double>& a, const vector<int>& phi3600,
                     double d, double thetaRad, double Pmax){
    double re = 0, im = 0;
    for (int i = 0; i < N; i++){
        double ang = phi3600[i] * PI / 1800.0 + 2.0 * PI * d * i * cos(thetaRad);
        re += a[i] * cos(ang);
        im += a[i] * sin(ang);
    }
    return (re * re + im * im) / (Pmax * Pmax);
}

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;

    int N = 8 + (int)llround(f * 56.0);     // 8..64
    int M = 16 + (int)llround(f * 164.0);   // 16..180
    int D1000 = 480 + (testId % 3) * 20;    // 480/500/520

    bool tapered = (testId >= 4);
    vector<double> a(N);
    for (int i = 0; i < N; i++){
        int amp1000;
        if (!tapered) amp1000 = 1000;
        else amp1000 = 400 + rnd.next(0, 600);   // 400..1000
        a[i] = amp1000 / 1000.0;
    }
    double Pmax = 0; for (double v : a) Pmax += v;
    double d = D1000 / 1000.0;

    // angle grid: M points evenly over [0,180] degrees
    vector<int> ang(M);
    for (int m = 0; m < M; m++)
        ang[m] = (int)llround((180000.0 * m) / (M - 1));

    vector<double> tgt(M, 0.0);
    int planted = (testId == 6 || testId == 8) ? 1 : 0;
    int nLobes = (testId <= 2) ? 1 : (testId <= 5 ? 2 + (testId % 2) : 2 + (testId % 3));

    if (planted){
        // hidden random phase-only excitation -> target = its own realized pattern
        vector<int> hidden(N);
        for (int i = 0; i < N; i++) hidden[i] = rnd.next(0, 3599);
        for (int m = 0; m < M; m++){
            double th = ang[m] * PI / 180000.0;
            tgt[m] = AFpow(N, a, hidden, d, th, Pmax);
        }
    } else {
        // sum of nLobes Gaussian-ish lobes at random off-broadside centers
        vector<double> centerDeg(nLobes), height(nLobes), sigma(nLobes);
        for (int L = 0; L < nLobes; L++){
            // avoid dead-center broadside (90) so "do-nothing" trivial is a poor fit
            double c = 15 + rnd.next(0, 150);
            if (c > 80 && c < 100) c += 35;
            if (c > 180) c -= 40;
            centerDeg[L] = c;
            height[L] = (L == 0) ? 1.0 : (0.45 + 0.05 * rnd.next(0, 8));  // secondary lobes lower but sizable
            sigma[L] = 6.0 + rnd.next(0, 6);
        }
        for (int m = 0; m < M; m++){
            double thDeg = ang[m] / 1000.0;
            double val = 0;
            for (int L = 0; L < nLobes; L++){
                double dd = thDeg - centerDeg[L];
                val = max(val, height[L] * exp(-(dd * dd) / (2 * sigma[L] * sigma[L])));
            }
            tgt[m] = val;
        }
    }

    // weights: base, elevated at hard-null angles (assigned below)
    vector<int> w(M, 15);

    // hard-null indices: pick K angles with LOW target value, spread across the grid,
    // biased into gaps between lobes / pattern skirts (low tgt is a good proxy).
    int K;
    if (testId == 1) K = 0;
    else K = min(M, 1 + (testId - 1));   // 2..10, capped at 10 by problem constraints
    K = min(K, 10);
    vector<int> lowIdx;
    for (int m = 0; m < M; m++) if (tgt[m] < 0.12) lowIdx.push_back(m);
    for (int i = (int)lowIdx.size() - 1; i > 0; i--) swap(lowIdx[i], lowIdx[rnd.next(0, i)]);
    vector<int> nullIdx;
    for (int i = 0; i < (int)lowIdx.size() && (int)nullIdx.size() < K; i++) nullIdx.push_back(lowIdx[i]);
    K = (int)nullIdx.size();
    for (int idx : nullIdx){ tgt[idx] = 0.0; w[idx] = 60; }

    double thresh = planted ? 0.06 : 0.05;   // planted patterns are tighter/cleaner

    // ---- emit ----
    printf("%d %d %d\n", N, M, D1000);
    for (int i = 0; i < N; i++) printf("%d%c", (int)llround(a[i] * 1000.0), i + 1 == N ? '\n' : ' ');
    for (int m = 0; m < M; m++) printf("%d%c", ang[m], m + 1 == M ? '\n' : ' ');
    for (int m = 0; m < M; m++){
        int t = (int)llround(tgt[m] * 10000.0);
        t = max(0, min(10000, t));
        printf("%d%c", t, m + 1 == M ? '\n' : ' ');
    }
    for (int m = 0; m < M; m++) printf("%d%c", w[m], m + 1 == M ? '\n' : ' ');
    printf("%d\n", K);
    for (int i = 0; i < K; i++) printf("%d%c", nullIdx[i] + 1, i + 1 == K ? '\n' : ' ');
    if (K == 0) printf("\n");
    printf("%d\n", (int)llround(thresh * 10000.0));
    return 0;
}
