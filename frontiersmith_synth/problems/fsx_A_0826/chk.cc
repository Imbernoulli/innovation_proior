#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// chk.cc -- "Biogas Villages: Digesters versus Pipelines" (concave-consolidation-siting).
// Minimization. M farms each supply waste; participant chooses K digester sites (ANY
// integer coordinates in [0,100000]) and an assignment of every farm to one site. A
// digester serving aggregate capacity cap costs A + Bc*cap^0.6 (concave -> economies of
// scale reward fewer, bigger digesters). Farm i shipping to a digester at distance d pays
// L*w_i*d pipeline loss (linear -> penalises consolidating far away). F = sum(digester
// costs) + sum(pipeline loss).
// Baseline B = a demand-blind UNIFORM GRID of K0=round(sqrt(M)) digesters over the farms'
// bounding box, each farm sent to its nearest grid digester -- the "just spread digesters
// evenly, ignore capacity economics and the planted cluster structure" naive reference.

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int M = inf.readInt();
    long long A = inf.readLong();
    long long Bc = inf.readLong();
    long long Lm = inf.readLong();
    vector<long long> X(M + 1), Y(M + 1), W(M + 1);
    for (int i = 1; i <= M; i++) {
        X[i] = inf.readLong();
        Y[i] = inf.readLong();
        W[i] = inf.readLong();
    }
    double L = Lm / 1000.0;
    const long long COORD_MAX = 100000;

    auto pdist = [&](double x1, double y1, double x2, double y2) -> double {
        double dx = x1 - x2, dy = y1 - y2;
        return sqrt(dx * dx + dy * dy);
    };

    // ---- read participant output (strict feasibility) ----
    int K = ouf.readInt(1, M, "K");
    vector<long long> DX(K + 1), DY(K + 1);
    for (int j = 1; j <= K; j++) {
        DX[j] = ouf.readInt(0, COORD_MAX, "dx");
        DY[j] = ouf.readInt(0, COORD_MAX, "dy");
    }
    vector<int> assign(M + 1);
    for (int i = 1; i <= M; i++) assign[i] = ouf.readInt(1, K, "assign");
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens after output");

    vector<long long> cap(K + 1, 0);
    for (int i = 1; i <= M; i++) cap[assign[i]] += W[i];

    double F = 0.0;
    for (int j = 1; j <= K; j++) F += (double)A + (double)Bc * pow((double)cap[j], 0.6);
    for (int i = 1; i <= M; i++) {
        int j = assign[i];
        F += L * (double)W[i] * pdist((double)X[i], (double)Y[i], (double)DX[j], (double)DY[j]);
    }
    if (!isfinite(F) || F <= 0) quitf(_wa, "objective not finite/positive");

    // ---- baseline: uniform grid of K0=round(sqrt(M)) digesters over the bounding box,
    // nearest-digester assignment. Demand-blind and cost-blind (no economies-of-scale or
    // cluster reasoning at all) -- trivial.cpp reproduces this exactly.
    long long x0 = LLONG_MAX, x1 = LLONG_MIN, y0 = LLONG_MAX, y1 = LLONG_MIN;
    for (int i = 1; i <= M; i++) {
        x0 = min(x0, X[i]); x1 = max(x1, X[i]);
        y0 = min(y0, Y[i]); y1 = max(y1, Y[i]);
    }
    int K0 = (int)llround(sqrt((double)M));
    if (K0 < 1) K0 = 1;
    int cols = (int)ceil(sqrt((double)K0));
    if (cols < 1) cols = 1;
    int rows = (K0 + cols - 1) / cols;
    vector<double> bx, by;
    for (int r = 0; r < rows && (int)bx.size() < K0; r++)
        for (int c = 0; c < cols && (int)bx.size() < K0; c++) {
            double gx = x0 + ((2.0 * c + 1) * (x1 - x0)) / (2.0 * cols);
            double gy = y0 + ((2.0 * r + 1) * (y1 - y0)) / (2.0 * rows);
            bx.push_back(gx); by.push_back(gy);
        }
    int KB = (int)bx.size();
    vector<long long> bcap(KB, 0);
    vector<int> bassign(M + 1);
    for (int i = 1; i <= M; i++) {
        int best = 0; double bd = 1e300;
        for (int j = 0; j < KB; j++) {
            double dd = pdist((double)X[i], (double)Y[i], bx[j], by[j]);
            if (dd < bd) { bd = dd; best = j; }
        }
        bassign[i] = best;
        bcap[best] += W[i];
    }
    double B = 0.0;
    for (int j = 0; j < KB; j++) B += (double)A + (double)Bc * pow((double)bcap[j], 0.6);
    for (int i = 1; i <= M; i++) {
        int j = bassign[i];
        B += L * (double)W[i] * pdist((double)X[i], (double)Y[i], bx[j], by[j]);
    }
    if (!isfinite(B) || B <= 0) B = 1;

    double sc = min(1000.0, 100.0 * B / max(1.0, F));
    quitp(sc / 1000.0, "OK F=%.3f B=%.3f Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
