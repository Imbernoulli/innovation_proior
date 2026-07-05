// TIER: trivial
// Parity baseline: exactly the checker's reference construction -> ratio == 0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m, tol;
    scanf("%d %d %d", &n, &m, &tol);
    for (int i = 0; i < m; i++) { int u, v, w; scanf("%d %d %d", &u, &v, &w); }
    for (int i = 1; i <= n; i++) printf("%d%c", i & 1, i == n ? '\n' : ' ');
    return 0;
}
