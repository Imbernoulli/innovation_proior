// TIER: trivial
// Build a depot in every shop district: exactly the checker's reference
// construction, so this scores the calibration baseline (ratio ~ 0.1).
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m, D, r;
    if (scanf("%d %d %d %d", &n, &m, &D, &r) != 4) return 0;
    vector<int> cost(n + 1);
    for (int u = 1; u <= n; u++) scanf("%d", &cost[u]);
    for (int i = 0; i < m; i++) { int u, v, w; scanf("%d %d %d", &u, &v, &w); }
    vector<int> shops(D);
    for (int i = 0; i < D; i++) scanf("%d", &shops[i]);

    printf("%d\n", D);
    for (int d : shops) printf("%d\n", d);
    return 0;
}
