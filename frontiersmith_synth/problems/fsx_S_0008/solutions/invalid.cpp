// TIER: invalid
// Deliberately infeasible: assign every sensor to station 0 (unbalanced).
// Must score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    for (int e = 0; e < m; e++) { int u, v, w; scanf("%d %d %d", &u, &v, &w); }
    for (int i = 1; i <= n; i++) printf("0%c", i == n ? '\n' : ' ');
    return 0;
}
