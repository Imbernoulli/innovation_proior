// TIER: invalid
// Deliberately infeasible: names a non-existent person (index n+1) -> the
// checker's bounded read rejects it -> score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m, r, k;
    if (scanf("%d %d %d %d", &n, &m, &r, &k) != 4) return 0;
    for (int i = 0; i < m; i++) { int u, v; scanf("%d %d", &u, &v); }
    printf("1\n%d\n", n + 1); // out-of-range seed index
    return 0;
}
