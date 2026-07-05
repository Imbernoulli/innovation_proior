// TIER: trivial
// Baseline: put every collector on channel 1 -> F == B -> ratio ~ 0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m, K;
    scanf("%d %d %d", &n, &m, &K);
    for (int i = 0; i < m; i++) { int u, v, p, q; scanf("%d %d %d %d", &u, &v, &p, &q); }
    for (int i = 0; i < n; i++) printf("1%c", i + 1 < n ? ' ' : '\n');
    return 0;
}
