// TIER: invalid
// Deliberately infeasible: assigns every module to cryostat 0, violating the exact-half
// balance rule (n/2 required in cryostat 1). Must score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    for (int e = 0; e < m; e++) { int u, v, t, w; scanf("%d %d %d %d", &u, &v, &t, &w); }
    for (int i = 1; i <= n; i++) printf("0%c", i == n ? '\n' : ' ');
    return 0;
}
