// TIER: invalid
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m, K;
    if (scanf("%d %d %d", &n, &m, &K) != 3) return 0;
    for (int i = 0; i < n; i++) { int p, s; scanf("%d %d", &p, &s); }
    for (int i = 0; i < m; i++) { int u, v, w, g; scanf("%d %d %d %d", &u, &v, &w, &g); }
    // deliberately infeasible: channel 0 is out of range [1..K] -> scores 0
    for (int i = 0; i < n; i++) printf("0%c", i + 1 < n ? ' ' : '\n');
    return 0;
}
