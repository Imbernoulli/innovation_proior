// TIER: invalid
// Deliberately infeasible: dumps every car onto track 1, violating the upper
// balance band (count = n > U). Must score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    long long n; int k; long long m, lam, L, U;
    if (scanf("%lld %d %lld %lld %lld %lld", &n, &k, &m, &lam, &L, &U) != 6) return 0;
    for (long long i = 1; i <= n; i++)
        printf("1%c", i == n ? '\n' : ' ');
    return 0;
}
