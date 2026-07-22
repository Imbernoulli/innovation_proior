// TIER: greedy
// Coarse block round-robin: partition the wafer into a fixed 4x4 grid of super-blocks,
// sort each block's fields in simple raster order, and visit blocks round-robin (one field
// from block 1, one from block 2, ..., one from the last block, then repeat).  A very
// natural first idea once you're told "don't revisit a neighbourhood too soon": spread
// visits COARSELY across the wafer.  It genuinely beats doing nothing, because it
// guarantees every early round touches every region instead of leaving it to luck -- but
// it only interleaves at ONE fixed scale.  Inside a block, consecutive rounds still emit
// raster-adjacent fields, and on any test where the natural neighbourhood scale differs
// from the fixed 4x4 partition (very fine dense clusters, or very long/thin regions), the
// single scale it picked is wrong and it degrades toward the same self-made hot spot.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N;
    if (scanf("%d", &N) != 1) return 0;
    double D, alpha, Q0;
    scanf("%lf %lf %lf", &D, &alpha, &Q0);
    vector<int> x(N), y(N), w(N);
    int minX = INT_MAX, maxX = INT_MIN, minY = INT_MAX, maxY = INT_MIN;
    for (int i = 0; i < N; i++) {
        scanf("%d %d %d", &x[i], &y[i], &w[i]);
        minX = min(minX, x[i]); maxX = max(maxX, x[i]);
        minY = min(minY, y[i]); maxY = max(maxY, y[i]);
    }
    if (N == 0) return 0;

    const int K = 3;  // fixed 3x3 super-block partition
    int blockW = max(1, (maxX - minX + 1 + K - 1) / K);
    int blockH = max(1, (maxY - minY + 1 + K - 1) / K);

    vector<vector<int>> buckets(K * K);
    for (int i = 0; i < N; i++) {
        int bx = min(K - 1, (x[i] - minX) / blockW);
        int by = min(K - 1, (y[i] - minY) / blockH);
        buckets[by * K + bx].push_back(i);
    }
    for (auto& b : buckets)
        sort(b.begin(), b.end(), [&](int a, int c) {
            if (y[a] != y[c]) return y[a] < y[c];
            return x[a] < x[c];
        });

    vector<int> order;
    order.reserve(N);
    vector<int> ptr(K * K, 0);
    int remaining = N;
    while (remaining > 0) {
        for (int b = 0; b < K * K; b++) {
            if (ptr[b] < (int)buckets[b].size()) {
                order.push_back(buckets[b][ptr[b]]);
                ptr[b]++;
                remaining--;
            }
        }
    }

    for (int id : order) printf("%d ", id + 1);
    printf("\n");
    return 0;
}
