// TIER: invalid
// Deliberately infeasible: assign every hive to yard 0. The yards are unbalanced
// (yard 0 holds all n hives, not n/k), so the checker must score this 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n; long long m; int k;
    scanf("%d %lld %d", &n, &m, &k);
    for (long long e = 0; e < m; e++) {
        int u, v; long long w;
        scanf("%d %d %lld", &u, &v, &w);
    }
    for (int i = 1; i <= n; i++)
        printf("0%c", i == n ? '\n' : ' ');
    return 0;
}
