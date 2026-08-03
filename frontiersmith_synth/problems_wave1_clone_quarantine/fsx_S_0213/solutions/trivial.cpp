// TIER: trivial
// Do-nothing baseline: seed every pool with niche 1. All channels monochromatic
// -> F == B -> ratio 0.1 exactly.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m, C;
    if (scanf("%d %d %d", &n, &m, &C) != 3) return 0;
    for (int i = 0; i < m; i++) { int u, v, w; scanf("%d %d %d", &u, &v, &w); }
    for (int i = 0; i < n; i++) printf("1\n");
    return 0;
}
