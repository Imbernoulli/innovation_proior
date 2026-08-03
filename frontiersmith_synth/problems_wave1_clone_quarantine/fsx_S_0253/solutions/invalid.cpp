// TIER: invalid
// Deliberately infeasible: clear EVERY corridor. This blows the clearing budget (and
// disconnects the den from the watering ground), so the checker must score it 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m, s, t; long long C;
    if (scanf("%d %d %d %d %lld", &n, &m, &s, &t, &C) != 5) return 0;
    for (int i = 0; i < m; i++) { int u, v, w, c; scanf("%d %d %d %d", &u, &v, &w, &c); }
    printf("%d\n", m);
    for (int i = 1; i <= m; i++) printf("%d\n", i);
    return 0;
}
