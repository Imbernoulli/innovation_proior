// TIER: invalid
// Deliberately infeasible: prints an out-of-range label (2) for zone 1, so the
// checker's bounded read rejects it -> score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m, slack;
    if (scanf("%d %d %d", &n, &m, &slack) != 3) return 0;
    for (int i = 0; i < m; i++) { int u, v, w; scanf("%d %d %d", &u, &v, &w); }
    printf("2"); // invalid label
    for (int i = 2; i <= n; i++) printf(" 0");
    printf("\n");
    return 0;
}
