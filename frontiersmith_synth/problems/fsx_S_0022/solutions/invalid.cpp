// TIER: invalid
// Deliberately infeasible: emits an out-of-range regime value (2) for every
// gate, so the checker's bounded read rejects it -> score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    for (int j = 0; j < m; j++) {
        int w, k; scanf("%d %d", &w, &k);
        for (int i = 0; i < k; i++) { int c; scanf("%d", &c); }
    }
    for (int g = 1; g <= n; g++) printf(g == 1 ? "2" : " 2");
    printf("\n");
    return 0;
}
