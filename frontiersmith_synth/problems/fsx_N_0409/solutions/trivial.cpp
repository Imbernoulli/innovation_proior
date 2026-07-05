// TIER: trivial
// Round-robin baseline: car i -> track ((i-1) mod k)+1. Exactly the checker's
// internal baseline B, so this scores ratio ~= 0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    long long n; int k; long long m, lam, L, U;
    if (scanf("%lld %d %lld %lld %lld %lld", &n, &k, &m, &lam, &L, &U) != 6) return 0;
    for (long long i = 1; i <= n; i++)
        printf("%lld%c", ((i - 1) % k) + 1, i == n ? '\n' : ' ');
    return 0;
}
