// TIER: invalid
// Deliberately infeasible: output both endpoints of a conflict edge (or an
// out-of-range index if there are no edges) -> violates independence -> score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    vector<long long> w(n + 1);
    for (int i = 1; i <= n; i++) scanf("%lld", &w[i]);
    int fu = -1, fv = -1;
    for (int i = 0; i < m; i++) {
        int u, v; scanf("%d %d", &u, &v);
        if (fu < 0) { fu = u; fv = v; }
    }
    if (fu >= 0) {
        // both endpoints of a real conflict edge: infeasible independent set
        printf("2\n%d\n%d\n", fu, fv);
    } else {
        // no edges at all: emit an out-of-range index
        printf("1\n%d\n", n + 1);
    }
    return 0;
}
