// TIER: trivial
#include <bits/stdc++.h>
using namespace std;

// Demand-blind uniform GRID of K0=round(sqrt(M)) digesters over the farms' bounding box,
// each farm sent to its nearest grid digester. Matches the checker's baseline B exactly
// (same construction, same tie-breaking), so this always scores ratio ~= 0.1.
int main() {
    int M; long long A, Bc, Lm;
    if (!(cin >> M >> A >> Bc >> Lm)) return 0;
    vector<long long> X(M + 1), Y(M + 1), W(M + 1);
    for (int i = 1; i <= M; i++) cin >> X[i] >> Y[i] >> W[i];

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
    int K = (int)bx.size();
    vector<int> assign(M + 1);
    for (int i = 1; i <= M; i++) {
        int best = 0; double bd = 1e300;
        for (int j = 0; j < K; j++) {
            double dx = (double)X[i] - bx[j], dy = (double)Y[i] - by[j];
            double dd = sqrt(dx * dx + dy * dy);
            if (dd < bd) { bd = dd; best = j; }
        }
        assign[i] = best + 1;
    }

    cout << K << "\n";
    for (int j = 0; j < K; j++) cout << (long long)llround(bx[j]) << " " << (long long)llround(by[j]) << "\n";
    for (int i = 1; i <= M; i++) cout << assign[i] << (i < M ? ' ' : '\n');
    return 0;
}
