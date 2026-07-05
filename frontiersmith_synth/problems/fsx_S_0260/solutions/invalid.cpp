// TIER: invalid
// Deliberately infeasible: put every wagon in Bowl 0 -> bowls maximally unbalanced -> score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m, tol;
    scanf("%d %d %d", &n, &m, &tol);
    for (int i = 0; i < m; i++) { int u, v, w; scanf("%d %d %d", &u, &v, &w); }
    for (int i = 1; i <= n; i++) printf("0%c", i == n ? '\n' : ' ');
    return 0;
}
