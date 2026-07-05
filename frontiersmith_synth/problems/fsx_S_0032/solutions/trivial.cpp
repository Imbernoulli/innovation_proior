// TIER: trivial
// Replicates the judge's balance baseline: walk performers in input order and
// always place the next on the currently lighter stage (ties -> North). By
// construction its cross-plaza flow equals B, so it scores ratio ~0.1.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main() {
    int n, m; ll tau;
    if (scanf("%d %d %lld", &n, &m, &tau) != 3) return 0;
    vector<int> a(n + 1);
    for (int i = 1; i <= n; i++) scanf("%d", &a[i]);
    for (int e = 0; e < m; e++) { int u, v, w; scanf("%d %d %d", &u, &v, &w); }
    ll T0 = 0, T1 = 0;
    for (int i = 1; i <= n; i++) {
        if (T0 <= T1) { printf("0\n"); T0 += a[i]; }
        else          { printf("1\n"); T1 += a[i]; }
    }
    return 0;
}
