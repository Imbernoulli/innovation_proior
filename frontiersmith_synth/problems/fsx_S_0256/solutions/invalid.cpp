// TIER: invalid
// Deliberately infeasible: emits a comb-cell (delivery) with no matching flower patch,
// violating the pickup-before-delivery / both-or-neither rule -> must score 0.
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
    printf("1\n");
    printf("D 1\n"); // delivery without pickup -> infeasible
    return 0;
}
