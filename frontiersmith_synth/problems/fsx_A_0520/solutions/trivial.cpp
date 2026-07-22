// TIER: trivial
// Demand-blind uniform GRID baseline: spread the P depots evenly over the address bounding
// box, ignoring seasons entirely -- exactly the construction the checker measures as B, so
// this scores ~0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, P, K;
    if (scanf("%d %d %d", &N, &P, &K) != 3) return 0;
    long long x0 = LLONG_MAX, x1 = LLONG_MIN, y0 = LLONG_MAX, y1 = LLONG_MIN;
    for (int i = 0; i < N; i++) {
        long long x, y, w;
        scanf("%lld %lld", &x, &y);
        for (int k = 0; k < K; k++) scanf("%lld", &w);
        x0 = min(x0, x); x1 = max(x1, x); y0 = min(y0, y); y1 = max(y1, y);
    }
    long long cols = (long long)ceil(sqrt((double)P));
    if (cols < 1) cols = 1;
    long long rows = (P + cols - 1) / cols;
    long long placed = 0;
    for (long long r = 0; r < rows && placed < P; r++)
        for (long long c = 0; c < cols && placed < P; c++) {
            long long gx = x0 + ((2*c + 1) * (x1 - x0)) / (2*cols);
            long long gy = y0 + ((2*r + 1) * (y1 - y0)) / (2*rows);
            printf("%lld %lld\n", gx, gy);
            placed++;
        }
    return 0;
}
