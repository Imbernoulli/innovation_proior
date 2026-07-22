// TIER: invalid
// Deliberately infeasible: pile every animal onto shelf 1, blowing past pageCap.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m, pageCap, L, T;
    if (scanf("%d %d %d %d %d", &n, &m, &pageCap, &L, &T) != 5) return 0;
    for (int i = 0; i < m; i++) { int u, v; scanf("%d %d", &u, &v); }
    for (int i = 0; i < T; i++) { int x; scanf("%d", &x); }
    for (int i = 1; i <= n; i++) printf("1\n");
    return 0;
}
