// TIER: invalid
// Deliberately infeasible: schedules an out-of-range target index (n+1) -> checker rejects -> 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    for (int i = 1; i <= n; i++) { long long w; scanf("%lld", &w); }
    for (int e = 0; e < m; e++) { int u, v; scanf("%d %d", &u, &v); }
    printf("1\n%d\n", n + 1); // out-of-range target index
    return 0;
}
