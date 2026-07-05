// TIER: invalid
// Deliberately infeasible: put both endpoints of a conflicting pair on the
// slate (violates independence) -> must score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    vector<int> w(n + 1);
    for (int i = 1; i <= n; i++) scanf("%d", &w[i]);
    int fu = -1, fv = -1;
    for (int i = 0; i < m; i++) {
        int u, v; scanf("%d %d", &u, &v);
        if (fu < 0) { fu = u; fv = v; }
    }
    if (fu > 0) {
        printf("2\n%d %d\n", fu, fv);   // conflicting pair -> infeasible
    } else {
        printf("1\n%d\n", n + 1);       // no edges: out-of-range index
    }
    return 0;
}
