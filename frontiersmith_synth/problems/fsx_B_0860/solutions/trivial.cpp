// TIER: trivial
// The checker's own baseline: identity run order (p_j = j), no tip changes, no
// recalibrations at all. Always feasible since deadline_i >= i is guaranteed.
// Reproduces B exactly, so this scores ratio ~= 0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, K1, K2, GAP_COEF, DRIFT_COEF, BOOST;
    scanf("%d %d %d %d %d %d", &n, &K1, &K2, &GAP_COEF, &DRIFT_COEF, &BOOST);
    for (int i = 1; i <= n; i++) {
        long long c; int st, dl;
        scanf("%lld %d %d", &c, &st, &dl);
    }
    for (int i = 1; i <= n; i++) printf("%d%c", i, i == n ? '\n' : ' ');
    for (int i = 1; i <= n; i++) printf("0%c", i == n ? '\n' : ' ');
    for (int i = 1; i <= n; i++) printf("0%c", i == n ? '\n' : ' ');
    return 0;
}
