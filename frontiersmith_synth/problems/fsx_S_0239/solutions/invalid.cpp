// TIER: invalid
// Deliberately infeasible: activate BOTH endpoints of a conflict edge.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    vector<int> w(n + 1);
    for (int i = 1; i <= n; i++) scanf("%d", &w[i]);
    int fu = -1, fv = -1;
    for (int i = 0; i < m; i++) {
        int u, v; scanf("%d %d", &u, &v);
        if (i == 0) { fu = u; fv = v; }
    }
    if (fu == -1) { printf("0\n"); return 0; }
    // output both endpoints of a conflict -> not an independent set -> score 0
    printf("2\n%d %d\n", fu, fv);
    return 0;
}
