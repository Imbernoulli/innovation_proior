// TIER: invalid
// Deliberately infeasible: the pair[] value printed for vertex 1 of every
// arena is n+50, outside the required [0,n] range, which the checker's
// bounded read must reject.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int K;
    scanf("%d", &K);
    for (int a = 0; a < K; a++) {
        int n, m, first;
        scanf("%d %d %d", &n, &m, &first);
        for (int j = 0; j < m; j++) { int u, v; scanf("%d %d", &u, &v); }
        for (int v = 1; v <= n; v++) {
            if (v == 1) printf("%d ", n + 50);
            else printf("%d ", 0);
        }
        printf("\n");
        for (int v = 1; v <= n; v++) printf("%d ", v);
        printf("\n");
    }
    return 0;
}
