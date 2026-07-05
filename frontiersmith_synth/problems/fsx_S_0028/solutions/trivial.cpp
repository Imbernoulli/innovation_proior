// TIER: trivial
// Serialized plan: serve each job fully in index order. This is exactly the
// checker's baseline construction, so it scores ratio = 0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, Q;
    scanf("%d %d", &N, &Q);
    long long x0, y0; scanf("%lld %lld", &x0, &y0);
    for (int i = 1; i <= N; i++) {
        long long a,b,c,d; int q;
        scanf("%lld %lld %lld %lld %d", &a, &b, &c, &d, &q);
    }
    printf("%d\n", 2 * N);
    for (int i = 1; i <= N; i++) {
        printf("0 %d\n", i);
        printf("1 %d\n", i);
    }
    return 0;
}
