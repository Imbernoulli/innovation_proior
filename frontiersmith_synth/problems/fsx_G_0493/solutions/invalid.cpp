// TIER: invalid
// Issue every operation in cycle 0.  With any dependence edge u->v (L_u >= 1) this violates
// s_v >= s_u + L_u, and it also blows the issue width / unit caps / read-port budget.
// Deliberately infeasible -> must score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m, W, T, P;
    if (scanf("%d %d %d %d %d", &n, &m, &W, &T, &P) != 5) return 0;
    vector<int> cap(T + 1), rp(T + 1);
    for (int t = 1; t <= T; t++) scanf("%d", &cap[t]);
    for (int t = 1; t <= T; t++) scanf("%d", &rp[t]);
    for (int i = 1; i <= n; i++) { int a, b; scanf("%d %d", &a, &b); }
    for (int j = 0; j < m; j++) { int u, v; scanf("%d %d", &u, &v); }
    for (int i = 1; i <= n; i++) printf("0\n");
    return 0;
}
