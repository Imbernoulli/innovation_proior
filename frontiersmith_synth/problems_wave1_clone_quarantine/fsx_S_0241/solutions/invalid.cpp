// TIER: invalid
// Deliberately infeasible: powers down the inlet probe s itself -> checker rejects -> score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m, s, t, k;
    if (scanf("%d %d %d %d %d", &n, &m, &s, &t, &k) != 5) return 0;
    for (int i = 0; i < m; i++) { int u, v, w; scanf("%d %d %d", &u, &v, &w); }
    printf("1\n%d\n", s); // powering down the inlet is forbidden
    return 0;
}
