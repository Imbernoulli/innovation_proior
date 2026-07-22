// TIER: invalid
// Deliberately infeasible: claims a single upgrade whose cost blows through any
// possible budget, so the checker's budget check must reject it (score 0).
#include <bits/stdc++.h>
int main() {
    long long R, C, M, K, B;
    scanf("%lld %lld %lld %lld %lld", &R, &C, &M, &K, &B);
    printf("1\n0 1000000000\n");
    return 0;
}
