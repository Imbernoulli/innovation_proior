// TIER: trivial
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m, lo, hi;
    scanf("%d %d %d %d", &n, &m, &lo, &hi);
    for (int i = 0; i < m; i++) {
        int u, v, w;
        scanf("%d %d %d", &u, &v, &w);
    }
    int half = n / 2;
    for (int i = 1; i <= n; i++) {
        printf("%d ", (i <= half) ? 0 : 1);
    }
    printf("\n");
    return 0;
}
