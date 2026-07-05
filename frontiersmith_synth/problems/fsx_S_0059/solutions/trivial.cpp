// TIER: trivial
// Irrigate only the single highest-yield plot => F = B => ratio 0.1 (the checker baseline).
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m; long long W;
    if (scanf("%d %d %lld", &n, &m, &W) != 3) return 0;
    vector<long long> w(n + 1), d(n + 1);
    for (int i = 1; i <= n; i++) scanf("%lld", &w[i]);
    for (int i = 1; i <= n; i++) scanf("%lld", &d[i]);
    // edges irrelevant for a single plot
    int best = 1;
    for (int i = 2; i <= n; i++) if (w[i] > w[best]) best = i;
    printf("1\n%d\n", best);
    return 0;
}
