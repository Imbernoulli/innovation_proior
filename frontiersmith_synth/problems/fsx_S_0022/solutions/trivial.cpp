// TIER: trivial
// All-low-flow baseline: set every gate to 0. This reproduces the checker's
// internal baseline B, so F == B and the ratio is exactly 0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    for (int j = 0; j < m; j++) {
        int w, k; scanf("%d %d", &w, &k);
        for (int i = 0; i < k; i++) { int c; scanf("%d", &c); }
    }
    for (int g = 1; g <= n; g++) printf(g == 1 ? "0" : " 0");
    printf("\n");
    return 0;
}
