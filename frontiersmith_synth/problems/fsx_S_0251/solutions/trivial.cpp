// TIER: trivial
// Single-best baseline: schedule ONLY the highest-value target. F == B -> ratio 0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    int best = 1; long long bw = -1;
    for (int i = 1; i <= n; i++) {
        long long w; scanf("%lld", &w);
        if (w > bw) { bw = w; best = i; }
    }
    for (int e = 0; e < m; e++) { int u, v; scanf("%d %d", &u, &v); }
    printf("1\n%d\n", best);
    return 0;
}
