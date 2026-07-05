// TIER: invalid
// Deliberately infeasible: monitor BOTH endpoints of a recorded contact.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    for (int i = 1; i <= n; i++) { long long x; scanf("%lld", &x); }
    int fu = -1, fv = -1;
    for (int e = 0; e < m; e++) {
        int u, v; scanf("%d %d", &u, &v);
        if (fu < 0) { fu = u; fv = v; }
    }
    if (fu < 0) { // no edges: fall back to an out-of-range index to force infeasible
        printf("1\n%d\n", n + 1);
        return 0;
    }
    printf("2\n%d\n%d\n", fu, fv); // adjacent pair -> not independent -> score 0
    return 0;
}
