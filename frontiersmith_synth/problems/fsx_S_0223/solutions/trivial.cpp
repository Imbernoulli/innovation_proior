// TIER: trivial
// Input-order chain 1-2-...-n : exactly the checker's baseline B -> ratio ~= 0.1
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
    for (int i = 1; i < n; i++) printf("%d %d\n", i, i + 1);
    return 0;
}
