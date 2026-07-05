// TIER: trivial
// Build a watchtower on every cell -> always feasible, cost F = B, ratio 0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, M, r;
    if (scanf("%d %d %d", &N, &M, &r) != 3) return 0;
    for (int i = 1; i <= N; i++) { int c; scanf("%d", &c); }
    for (int e = 0; e < M; e++) { int u, v; scanf("%d %d", &u, &v); }
    printf("%d\n", N);
    for (int i = 1; i <= N; i++) printf("%d\n", i);
    return 0;
}
