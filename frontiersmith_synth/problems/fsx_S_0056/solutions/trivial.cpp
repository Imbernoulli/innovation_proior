// TIER: trivial
// Index split: zones 1..n/2 -> depot 0, rest -> depot 1. This is exactly the
// baseline the checker measures, so F == B -> ratio 0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m, slack;
    if (scanf("%d %d %d", &n, &m, &slack) != 3) return 0;
    for (int i = 0; i < m; i++) { int u, v, w; scanf("%d %d %d", &u, &v, &w); }
    for (int i = 1; i <= n; i++) printf("%d%c", (i <= n / 2) ? 0 : 1, i == n ? '\n' : ' ');
    return 0;
}
