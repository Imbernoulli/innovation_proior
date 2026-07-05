// TIER: invalid
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    vector<long long> w(n + 1);
    for (int i = 1; i <= n; i++) scanf("%lld", &w[i]);
    int fu = -1, fv = -1;
    for (int i = 0; i < m; i++) {
        int u, v; scanf("%d %d", &u, &v);
        if (i == 0) { fu = u; fv = v; }
    }
    // deliberately equip two CONFLICTING sites -> infeasible -> must score 0.
    if (m > 0) printf("2\n%d\n%d\n", fu, fv);
    else       printf("1\n%d\n", n + 1); // no edges: emit out-of-range index instead
    return 0;
}
