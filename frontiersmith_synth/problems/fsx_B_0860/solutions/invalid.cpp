// TIER: invalid
// Deliberately infeasible: the run order repeats sample id 1 for every position
// instead of being a permutation of 1..n.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, K1, K2, GAP_COEF, DRIFT_COEF, BOOST;
    scanf("%d %d %d %d %d %d", &n, &K1, &K2, &GAP_COEF, &DRIFT_COEF, &BOOST);
    for (int i = 1; i <= n; i++) {
        long long c; int st, dl;
        scanf("%lld %d %d", &c, &st, &dl);
    }
    for (int i = 1; i <= n; i++) printf("1%c", i == n ? '\n' : ' ');
    for (int i = 1; i <= n; i++) printf("0%c", i == n ? '\n' : ' ');
    for (int i = 1; i <= n; i++) printf("0%c", i == n ? '\n' : ' ');
    return 0;
}
