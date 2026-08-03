// TIER: invalid
// Deliberately infeasible: emits a niche index (0) outside 1..C -> checker rejects -> score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m, C;
    if (scanf("%d %d %d", &n, &m, &C) != 3) return 0;
    for (int i = 0; i < m; i++) { int u, v, w; scanf("%d %d %d", &u, &v, &w); }
    for (int i = 0; i < n; i++) printf("0\n"); // 0 is out of range (< 1)
    return 0;
}
