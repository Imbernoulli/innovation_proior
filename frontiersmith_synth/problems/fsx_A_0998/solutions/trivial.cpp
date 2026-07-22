// TIER: trivial
// Do-everything baseline: recruit every household as a seed grower. This
// reproduces the checker's internal baseline B exactly (cost = sum of all
// c_i) -> ratio ~0.1. Always feasible (every household is NEW from season
// 0 regardless of R, so acreage coverage is trivially 100% >= tau).
#include <bits/stdc++.h>
using namespace std;

int main(){
    int n, m, R, tau;
    if (scanf("%d %d %d %d", &n, &m, &R, &tau) != 4) return 0;
    for (int i = 0; i < n; i++){ int p, c, s; scanf("%d %d %d", &p, &c, &s); }
    for (int i = 0; i < m; i++){ int u, v, w; scanf("%d %d %d", &u, &v, &w); }
    printf("%d\n", n);
    for (int i = 1; i <= n; i++) printf("%d ", i);
    printf("\n");
    return 0;
}
