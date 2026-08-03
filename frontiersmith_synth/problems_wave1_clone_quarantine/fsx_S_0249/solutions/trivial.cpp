// TIER: trivial
// Do-nothing baseline: every rig on channel 1. F == B -> ratio 0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m, C;
    if (scanf("%d %d %d", &n, &m, &C) != 3) return 0;
    for (int i = 0; i < m; i++) { int u, v, p, q; scanf("%d %d %d %d", &u, &v, &p, &q); }
    for (int i = 0; i < n; i++) printf("1%c", i + 1 == n ? '\n' : ' ');
    return 0;
}
