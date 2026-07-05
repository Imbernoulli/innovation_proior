// TIER: invalid
// Deliberately infeasible: emit an out-of-range crew label (2) so the checker's
// bounded read rejects it -> score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    for (int e = 0; e < m; e++) { int u, v, w; if (scanf("%d %d %d", &u, &v, &w) != 3) return 0; }
    printf("2"); // invalid label
    for (int i = 2; i <= n; i++) printf(" 0");
    printf("\n");
    return 0;
}
