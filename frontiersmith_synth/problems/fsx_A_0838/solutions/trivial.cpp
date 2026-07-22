// TIER: trivial
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m, pageCap, L, T;
    if (scanf("%d %d %d %d %d", &n, &m, &pageCap, &L, &T) != 5) return 0;
    for (int i = 0; i < m; i++) { int u, v; scanf("%d %d", &u, &v); }
    for (int i = 0; i < T; i++) { int x; scanf("%d", &x); }

    for (int i = 1; i <= n; i++) {
        int p = 1 + (i - 1) / pageCap;
        printf("%d\n", p);
    }
    return 0;
}
