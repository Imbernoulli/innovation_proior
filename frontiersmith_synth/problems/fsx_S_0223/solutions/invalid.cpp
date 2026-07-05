// TIER: invalid
// Star from node 1 to every other node: node 1 gets degree n-1, which exceeds any
// capacity (c_i <= 4) -> infeasible -> must score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (scanf("%d", &n) != 1) return 0;
    for (int i = 1; i <= n; i++) {
        long long x, y, c;
        scanf("%lld %lld %lld", &x, &y, &c);
    }
    printf("%d\n", n - 1);
    for (int i = 2; i <= n; i++) printf("1 %d\n", i);
    return 0;
}
