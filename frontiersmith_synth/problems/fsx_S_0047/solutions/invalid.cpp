// TIER: invalid
// Deliberately infeasible: selects both endpoints of the first conflict pair, so the
// selection is NOT an independent set -> the checker must score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    for (int i = 1; i <= n; i++) { long long x; scanf("%lld", &x); }
    int fu = 1, fv = 2;
    for (int e = 0; e < m; e++) {
        int u, v;
        scanf("%d %d", &u, &v);
        if (e == 0) { fu = u; fv = v; }
    }
    printf("2\n%d %d\n", fu, fv); // two conflicting sites -> infeasible
    return 0;
}
