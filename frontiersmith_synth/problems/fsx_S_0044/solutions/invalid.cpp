// TIER: invalid
// Deliberately infeasible: dock every asteroid at refinery 0, so 0 asteroids are at
// refinery 1 (imbalanced) -> the checker must reject this and score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    for (int i = 0; i < m; i++) { int u, v, w; scanf("%d %d %d", &u, &v, &w); }
    for (int i = 1; i <= n; i++) printf("0\n");
    return 0;
}
