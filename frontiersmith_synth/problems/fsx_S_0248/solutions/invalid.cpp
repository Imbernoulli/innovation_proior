// TIER: invalid
// Deliberately infeasible: assigns every rack to the blue loop (0 racks on red),
// violating the equal-capacity rule -> checker rejects -> score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    for (int i = 0; i < m; i++) { int u, v, w; if (scanf("%d %d %d", &u, &v, &w) != 3) break; }
    for (int i = 1; i <= n; i++) printf("0\n");
    return 0;
}
