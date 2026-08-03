// TIER: invalid
// Deliberately infeasible: activate BOTH endpoints of the first interfering pair, which
// violates the conflict-free constraint and must score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M;
    scanf("%d %d", &N, &M);
    vector<long long> w(N + 1);
    for (int j = 1; j <= N; j++) scanf("%lld", &w[j]);
    int fu = -1, fv = -1;
    for (int i = 0; i < M; i++) {
        int u, v; scanf("%d %d", &u, &v);
        if (i == 0) { fu = u; fv = v; }
    }
    if (M >= 1) {
        printf("2\n%d %d\n", fu, fv);      // interfering pair -> infeasible
    } else {
        printf("1\n%d\n", N + 1);          // out-of-range fallback
    }
    return 0;
}
