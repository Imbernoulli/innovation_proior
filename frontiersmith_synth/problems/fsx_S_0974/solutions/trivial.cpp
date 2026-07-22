// TIER: trivial
// Build nothing anywhere: h_i = 0 for every segment. This is exactly the checker's own
// do-nothing baseline construction, so it scores ratio ~= 0.1. Always feasible (spends 0
// of the budget, and 0 is always within [0, Hmax_i]).
#include <bits/stdc++.h>
using namespace std;

int main() {
    long long N, T, Budget;
    if (scanf("%lld %lld %lld", &N, &T, &Budget) != 3) return 0;
    for (long long i = 0; i < N; i++) {
        long long a, b, c, d, e;
        scanf("%lld %lld %lld %lld %lld", &a, &b, &c, &d, &e);
    }
    for (long long t = 0; t < T; t++) {
        long long x;
        scanf("%lld", &x);
    }
    for (long long i = 0; i < N; i++) printf("0%c", (i + 1 == N) ? '\n' : ' ');
    return 0;
}
