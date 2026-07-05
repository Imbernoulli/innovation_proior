// TIER: trivial
// Reference id-order split: modules 1..n/2 -> cryostat 0, rest -> cryostat 1.
// This is exactly the baseline the checker measures, so it scores ratio ~0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    for (int e = 0; e < m; e++) { int u, v, t, w; scanf("%d %d %d %d", &u, &v, &t, &w); }
    for (int i = 1; i <= n; i++) printf("%d%c", (i <= n / 2) ? 0 : 1, i == n ? '\n' : ' ');
    return 0;
}
