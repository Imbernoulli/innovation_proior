// TIER: invalid
// Deliberately infeasible: emits channel 0 (out of the valid 1..K range) -> score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m, K;
    scanf("%d %d %d", &n, &m, &K);
    for (int i = 0; i < m; i++) { int u, v, p, q; scanf("%d %d %d %d", &u, &v, &p, &q); }
    for (int i = 0; i < n; i++) printf("0%c", i + 1 < n ? ' ' : '\n');
    return 0;
}
