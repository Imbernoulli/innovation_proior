// TIER: trivial
// Do-nothing baseline: put every depot on circuit 1 -> objective F == B -> ratio ~0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m, C;
    if (scanf("%d %d %d", &n, &m, &C) != 3) return 0;
    for (int i = 0; i < m; i++) {
        int u, v, w;
        scanf("%d %d %d", &u, &v, &w);
    }
    for (int i = 0; i < n; i++) printf("%d ", 1);
    printf("\n");
    return 0;
}
