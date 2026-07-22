// TIER: invalid
// Deliberately infeasible: emits an out-of-range depot coordinate (-1), which the checker
// rejects via readInt(0,100000). Must score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, P, K;
    if (scanf("%d %d %d", &N, &P, &K) != 3) return 0;
    long long x, y, w;
    for (int i = 0; i < N; i++) {
        scanf("%lld %lld", &x, &y);
        for (int k = 0; k < K; k++) scanf("%lld", &w);
    }
    for (int p = 0; p < P; p++) printf("%d %d\n", -1, -1);
    return 0;
}
