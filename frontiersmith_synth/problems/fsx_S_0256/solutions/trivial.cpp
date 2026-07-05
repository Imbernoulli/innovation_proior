// TIER: trivial
// Serve every task in index order: reproduces the reference baseline B -> ratio ~= 0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N;
    if (scanf("%d", &N) != 1) return 0;
    long long Hx, Hy; scanf("%lld %lld", &Hx, &Hy);
    for (int i = 0; i < N; i++) {
        long long a, b, c, d, w;
        scanf("%lld %lld %lld %lld %lld", &a, &b, &c, &d, &w);
    }
    printf("%d\n", 2 * N);
    for (int i = 1; i <= N; i++) printf("P %d\nD %d\n", i, i);
    return 0;
}
