// TIER: invalid
// Deliberately infeasible: claim one beacon but give an out-of-range index,
// which the checker rejects -> score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, r;
    if (scanf("%d %d", &N, &r) != 2) return 0;
    for (int i = 1; i <= N; i++) {
        long long a, b, c, d;
        scanf("%lld %lld %lld %lld", &a, &b, &c, &d);
    }
    printf("1\n%d\n", N + 1); // index out of [1,N] -> feasibility violation
    return 0;
}
