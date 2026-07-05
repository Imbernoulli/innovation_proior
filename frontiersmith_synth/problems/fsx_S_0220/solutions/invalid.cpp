// TIER: invalid
// Deliberately infeasible: visit a delivery node without its pickup (precedence
// violation), which the grader must reject with score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (scanf("%d", &n) != 1) return 0;
    long long dx, dy; scanf("%lld %lld", &dx, &dy);
    for (int i = 1; i <= n; i++) {
        long long a, b, c, d, p;
        scanf("%lld %lld %lld %lld %lld", &a, &b, &c, &d, &p);
    }
    // Visit delivery of relay 1 (node n+1) with no pickup -> infeasible.
    printf("1\n%d\n", n + 1);
    return 0;
}
