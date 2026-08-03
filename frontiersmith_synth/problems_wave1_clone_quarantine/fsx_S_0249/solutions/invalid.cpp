// TIER: invalid
// Deliberately infeasible: assigns rig 1 an out-of-range channel (C+1) -> checker rejects -> score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m, C;
    if (scanf("%d %d %d", &n, &m, &C) != 3) return 0;
    for (int i = 0; i < m; i++) { int u, v, p, q; scanf("%d %d %d %d", &u, &v, &p, &q); }
    printf("%d", C + 1); // out-of-range channel for rig 1
    for (int i = 1; i < n; i++) printf(" 1");
    printf("\n");
    return 0;
}
