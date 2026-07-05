// TIER: invalid
// Deliberately infeasible: emit circuit 0 for every depot (0 is out of range 1..C).
// The checker's bounded read rejects it -> score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m, C;
    if (scanf("%d %d %d", &n, &m, &C) != 3) return 0;
    for (int i = 0; i < m; i++) {
        int u, v, w;
        scanf("%d %d %d", &u, &v, &w);
    }
    for (int i = 0; i < n; i++) printf("%d ", 0);
    printf("\n");
    return 0;
}
