// TIER: trivial
// Alternating assignment x_i = i mod 2 -- exactly the checker's internal baseline,
// so F == B and the score is the calibration point 0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m, D;
    if (scanf("%d %d %d", &n, &m, &D) != 3) return 0;
    for (int i = 0; i < m; i++) { int u, v, w; scanf("%d %d %d", &u, &v, &w); }
    for (int i = 1; i <= n; i++) printf("%d\n", i & 1);
    return 0;
}
