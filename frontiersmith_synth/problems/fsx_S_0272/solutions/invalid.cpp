// TIER: invalid
// Deliberately infeasible: lists a crew id that does not exist (n+1) -> checker rejects -> 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m, L;
    if (scanf("%d %d %d", &n, &m, &L) != 3) return 0;
    for (int i = 0; i < m; i++) { int u, v, w; scanf("%d %d %d", &u, &v, &w); }
    printf("1\n%d\n", n + 1);   // out-of-range crew id
    return 0;
}
