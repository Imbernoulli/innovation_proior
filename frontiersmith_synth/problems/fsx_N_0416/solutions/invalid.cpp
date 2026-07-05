// TIER: invalid
// Deliberately infeasible: cut EVERY line. Total cost = sum c_i > B (budget is < sum c_i by
// construction) and it also islands t, so the checker must reject this and score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m, s, t; long long B;
    if (scanf("%d %d %d %d %lld", &n, &m, &s, &t, &B) != 5) return 0;
    for (int e = 0; e < m; e++) { int u, v, r, c; scanf("%d %d %d %d", &u, &v, &r, &c); }
    printf("%d\n", m);
    for (int e = 1; e <= m; e++) printf("%d%c", e, e == m ? '\n' : ' ');
    return 0;
}
