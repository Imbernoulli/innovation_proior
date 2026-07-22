// TIER: strong
// Reframe the tour problem as DISCREPANCY minimization.  Build one spatially-coherent
// list L (row-major raster order over the fields), then visit L's positions in
// generalized bit-reversal (van der Corput) order of their index.  Bit-reversal visits
// dyadically bisect the largest unvisited gap in index space at every step, so consecutive
// EXPOSURE steps are maximally spread apart along the spatial ordering L -- exactly what
// lets diffused heat decay before a spatial neighbour is revisited.  This is a genuine
// reformulation (index-discrepancy, not tour length) and needs no knowledge of D/alpha/Q0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N;
    if (scanf("%d", &N) != 1) return 0;
    double D, alpha, Q0;
    scanf("%lf %lf %lf", &D, &alpha, &Q0);
    vector<int> x(N), y(N), w(N);
    for (int i = 0; i < N; i++) scanf("%d %d %d", &x[i], &y[i], &w[i]);
    if (N == 0) return 0;

    // ---- L: spatially coherent raster order (row-major, no serpentine) ----
    vector<int> L(N);
    for (int i = 0; i < N; i++) L[i] = i;
    sort(L.begin(), L.end(), [&](int a, int b) {
        if (y[a] != y[b]) return y[a] < y[b];
        return x[a] < x[b];
    });

    // ---- generalized bit-reversal permutation of [0, N) ----
    int bits = 0;
    long long M = 1;
    while (M < N) { M <<= 1; bits++; }
    vector<int> sigma;
    sigma.reserve(N);
    for (long long i = 0; i < M; i++) {
        long long r = 0;
        long long v = i;
        for (int b = 0; b < bits; b++) {
            r = (r << 1) | (v & 1);
            v >>= 1;
        }
        if (r < N) sigma.push_back((int)r);
    }

    for (int t = 0; t < N; t++) printf("%d ", L[sigma[t]] + 1);
    printf("\n");
    return 0;
}
