// TIER: invalid
// Deliberately infeasible: labels everyone 0, blowing the balance constraint
// (hi is always < n by construction), which must score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m, lo, hi;
    scanf("%d %d %d %d", &n, &m, &lo, &hi);
    for (int i = 0; i < m; i++) {
        int u, v, w;
        scanf("%d %d %d", &u, &v, &w);
    }
    for (int i = 1; i <= n; i++) printf("0 ");
    printf("\n");
    return 0;
}
