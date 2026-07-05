// TIER: invalid
// Deliberately INFEASIBLE: lights both endpoints of a conflicting pair, so the
// independence check fails and the score is 0. (If there are no edges at all,
// prints an out-of-range index instead, which is also rejected.)
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    for (int i = 1; i <= n; i++) { int x; scanf("%d", &x); (void)x; }
    if (m <= 0) {
        // no conflicts: emit an out-of-range ride index -> rejected.
        printf("1\n%d\n", n + 1);
        return 0;
    }
    int u = -1, v = -1;
    for (int i = 0; i < m; i++) {
        int a, b; scanf("%d %d", &a, &b);
        if (i == 0) { u = a; v = b; }
    }
    printf("2\n%d %d\n", u, v);
    return 0;
}
