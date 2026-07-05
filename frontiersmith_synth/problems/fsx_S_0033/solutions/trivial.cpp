// TIER: trivial
// Channel-cycling baseline: gallery g gets channel ((g-1) mod k)+1.
// This is exactly the checker's internal baseline -> ratio 0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, k, m;
    if (scanf("%d %d %d", &n, &k, &m) != 3) return 0;
    for (int i = 0; i < m; i++) { int u, v, w; if (scanf("%d %d %d", &u, &v, &w) != 3) break; }
    for (int g = 1; g <= n; g++) printf("%d%c", ((g - 1) % k) + 1, g == n ? '\n' : ' ');
    return 0;
}
