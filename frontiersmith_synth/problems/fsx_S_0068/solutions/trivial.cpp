// TIER: trivial
// Index-block reference split: hive i -> yard (i-1)/(n/k). This is exactly the
// baseline the checker measures, so it scores ratio ~0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n; long long m; int k;
    scanf("%d %lld %d", &n, &m, &k);
    for (long long e = 0; e < m; e++) {
        int u, v; long long w;
        scanf("%d %d %lld", &u, &v, &w);
    }
    int s = n / k;
    for (int i = 1; i <= n; i++) {
        printf("%d%c", (i - 1) / s, i == n ? '\n' : ' ');
    }
    return 0;
}
