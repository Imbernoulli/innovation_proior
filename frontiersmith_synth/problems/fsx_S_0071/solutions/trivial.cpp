// TIER: trivial
// Baseline: select only the single most-profitable route. A single route is always
// an independent set, so F = B and the score is exactly the calibration point 0.1.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    int best = 1; ll bw = -1;
    for (int i = 1; i <= n; i++) { ll x; scanf("%lld", &x); if (x > bw) { bw = x; best = i; } }
    // edges are irrelevant for a single-vertex answer; skip them.
    printf("1\n%d\n", best);
    return 0;
}
