// TIER: trivial
// Single-best-site baseline: build only the highest-value station. This is exactly
// the checker's baseline B, so F = B and it scores the calibration ratio ~0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    int best = 1;
    long long bw = -1;
    for (int i = 1; i <= n; i++) {
        long long x;
        scanf("%lld", &x);
        if (x > bw) { bw = x; best = i; }
    }
    // edges are irrelevant for a single-site selection; ignore the rest of stdin.
    printf("1\n%d\n", best);
    return 0;
}
