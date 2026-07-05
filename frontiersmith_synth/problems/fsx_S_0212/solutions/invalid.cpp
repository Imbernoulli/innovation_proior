// TIER: invalid
// Deliberately infeasible: assign every cell to fleet A. The imbalance |n - 0| = n > D,
// so the checker rejects it -> score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m, D;
    if (scanf("%d %d %d", &n, &m, &D) != 3) return 0;
    for (int i = 0; i < m; i++) { int u, v, w; scanf("%d %d %d", &u, &v, &w); }
    for (int i = 1; i <= n; i++) printf("0\n");
    return 0;
}
