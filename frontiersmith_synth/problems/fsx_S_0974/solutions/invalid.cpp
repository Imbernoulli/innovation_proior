// TIER: invalid
// Deliberately infeasible for every test: prints a height at segment 1 that exceeds its
// Hmax by 5 (everything else 0). The checker's bounded range check must reject this and
// score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    long long N, T, Budget;
    if (scanf("%lld %lld %lld", &N, &T, &Budget) != 3) return 0;
    vector<long long> base(N + 1), Hmax(N + 1), cost(N + 1), value(N + 1), store(N + 1);
    for (long long i = 1; i <= N; i++)
        scanf("%lld %lld %lld %lld %lld", &base[i], &Hmax[i], &cost[i], &value[i], &store[i]);
    for (long long t = 0; t < T; t++) {
        long long x;
        scanf("%lld", &x);
    }
    printf("%lld", Hmax[1] + 5);
    for (long long i = 2; i <= N; i++) printf(" 0");
    printf("\n");
    return 0;
}
