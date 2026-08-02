// TIER: trivial
// Baseline construction: never use the reactive mirror rule, and fall back to
// claiming vertices in plain ascending index order. This is exactly the
// construction the checker itself uses as its internal baseline B, so this
// solution scores ratio ~= 0.1 by definition of the scoring formula.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int K;
    scanf("%d", &K);
    for (int a = 0; a < K; a++) {
        int n, m, first;
        scanf("%d %d %d", &n, &m, &first);
        for (int j = 0; j < m; j++) { int u, v; scanf("%d %d", &u, &v); }
        for (int v = 1; v <= n; v++) printf("%d ", 0);
        printf("\n");
        for (int v = 1; v <= n; v++) printf("%d ", v);
        printf("\n");
    }
    return 0;
}
