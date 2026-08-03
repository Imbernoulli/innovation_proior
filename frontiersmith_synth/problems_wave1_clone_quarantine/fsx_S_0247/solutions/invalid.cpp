// TIER: invalid
// Deliberately infeasible: prints a single segment to an out-of-range location index,
// which the checker rejects (bounded readInt). Must score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N;
    if (scanf("%d", &N) != 1) return 0;
    for (int i = 0; i < N; i++) {
        long long x, y, t, cap;
        scanf("%lld %lld %lld %lld", &x, &y, &t, &cap);
    }
    // one segment, second endpoint out of [1,N] -> feasibility failure -> score 0
    printf("1\n");
    printf("1 %d\n", N + 50);
    return 0;
}
