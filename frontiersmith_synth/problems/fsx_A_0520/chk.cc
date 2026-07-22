#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// chk.cc -- scorer for "Courier depots that survive every season". Minimization.
// Reads N addresses (coords + K seasonal demands), then P participant depot points.
// For each address, d_i = Manhattan distance to its NEAREST depot (season-independent).
// Season cost cost_k = sum_i w[i][k]*d_i ; participant objective F = max_k cost_k.
// Baseline B = worst-season cost of the SINGLE-HUB layout (all depots at the unweighted
// integer centroid floor(mean x), floor(mean y)). ratio = min(1, 0.1 * B / max(1,F)).

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int P = inf.readInt();
    int K = inf.readInt();

    vector<long long> X(N), Y(N);
    vector<vector<long long>> W(N, vector<long long>(K));
    for (int i = 0; i < N; i++) {
        X[i] = inf.readInt();
        Y[i] = inf.readInt();
        for (int k = 0; k < K; k++) W[i][k] = inf.readInt();
    }

    const long long C = 100000;

    // ---- read participant depots (strict feasibility) ----
    vector<long long> DX(P), DY(P);
    for (int p = 0; p < P; p++) {
        DX[p] = ouf.readInt(0, C, "dx");
        DY[p] = ouf.readInt(0, C, "dy");
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after the %d depots", P);

    // ---- nearest-depot Manhattan distance per address ----
    auto worstSeason = [&](const vector<long long>& dx, const vector<long long>& dy) -> long long {
        int PP = (int)dx.size();
        vector<long long> cost(K, 0);
        for (int i = 0; i < N; i++) {
            long long best = LLONG_MAX;
            for (int p = 0; p < PP; p++) {
                long long dd = llabs(X[i] - dx[p]) + llabs(Y[i] - dy[p]);
                if (dd < best) best = dd;
            }
            for (int k = 0; k < K; k++) cost[k] += W[i][k] * best;
        }
        long long f = 0;
        for (int k = 0; k < K; k++) f = max(f, cost[k]);
        return f;
    };

    long long F = worstSeason(DX, DY);

    // ---- baseline B: demand-blind uniform GRID of P depots over the address bounding box.
    // This is the naive "just spread depots evenly across the map, ignore the seasons"
    // reference. Deterministic integer construction (trivial.cpp reproduces it exactly).
    long long x0 = LLONG_MAX, x1 = LLONG_MIN, y0 = LLONG_MAX, y1 = LLONG_MIN;
    for (int i = 0; i < N; i++) { x0 = min(x0, X[i]); x1 = max(x1, X[i]); y0 = min(y0, Y[i]); y1 = max(y1, Y[i]); }
    long long cols = (long long)ceil(sqrt((double)P));
    if (cols < 1) cols = 1;
    long long rows = (P + cols - 1) / cols;
    vector<long long> bx, by;
    for (long long r = 0; r < rows && (long long)bx.size() < P; r++)
        for (long long c = 0; c < cols && (long long)bx.size() < P; c++) {
            long long gx = x0 + ((2*c + 1) * (x1 - x0)) / (2*cols);
            long long gy = y0 + ((2*r + 1) * (y1 - y0)) / (2*rows);
            bx.push_back(gx); by.push_back(gy);
        }
    long long B = worstSeason(bx, by);
    if (B <= 0) B = 1;

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
