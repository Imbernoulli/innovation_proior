// TIER: trivial
// Reference index split: racks 1..n/2 -> blue(0), racks n/2+1..n -> red(1).
// This is exactly the checker's internal baseline, so F == B -> ratio 0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    for (int i = 0; i < m; i++) { int u, v, w; if (scanf("%d %d %d", &u, &v, &w) != 3) break; }
    for (int i = 1; i <= n; i++) printf("%d\n", (i <= n / 2) ? 0 : 1);
    return 0;
}
