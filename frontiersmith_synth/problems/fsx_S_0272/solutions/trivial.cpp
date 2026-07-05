// TIER: trivial
// Contiguous-split baseline: crew 1..floor(n/2) -> Habitat A. This is exactly the
// construction the checker measures as B, so F == B and the ratio is 0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m, L;
    if (scanf("%d %d %d", &n, &m, &L) != 3) return 0;
    for (int i = 0; i < m; i++) { int u, v, w; scanf("%d %d %d", &u, &v, &w); }
    int h = n / 2;
    printf("%d\n", h);
    for (int i = 1; i <= h; i++) printf("%d\n", i);
    return 0;
}
