// TIER: invalid
// Deliberately infeasible: emits channel 0 (out of range 1..k) -> checker rejects -> score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, k, m;
    if (scanf("%d %d %d", &n, &k, &m) != 3) return 0;
    for (int i = 0; i < m; i++) { int u, v, w; if (scanf("%d %d %d", &u, &v, &w) != 3) break; }
    for (int g = 1; g <= n; g++) printf("0%c", g == n ? '\n' : ' '); // 0 is out of range
    return 0;
}
