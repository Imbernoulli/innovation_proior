// TIER: invalid
// Deliberately infeasible: irrigate both endpoints of a real conflict edge (violates runoff constraint).
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m; long long W;
    if (scanf("%d %d %lld", &n, &m, &W) != 3) return 0;
    vector<long long> w(n + 1), d(n + 1);
    for (int i = 1; i <= n; i++) scanf("%lld", &w[i]);
    for (int i = 1; i <= n; i++) scanf("%lld", &d[i]);
    int u = 1, v = 2;
    if (m >= 1) scanf("%d %d", &u, &v);
    // print a conflicting pair -> not an independent set -> must score 0
    printf("2\n%d\n%d\n", u, v);
    return 0;
}
