// TIER: invalid
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Deliberately infeasible: releases an absurdly large amount every period,
// far above any reservoir's capacity, so the very first token already
// violates 0 <= r_i(t) <= avail_i(t). Must score 0.
int main() {
    int K, T;
    scanf("%d %d", &K, &T);
    for (int i = 1; i <= K; i++) { ll a, b, c, d; scanf("%lld %lld %lld %lld", &a, &b, &c, &d); }
    for (int i = 1; i < K; i++) { ll x; scanf("%lld", &x); }
    for (int t = 1; t <= T; t++) { ll x; scanf("%lld", &x); }
    for (int i = 1; i <= K; i++)
        for (int t = 1; t <= T; t++) { ll x; scanf("%lld", &x); }

    for (int t = 1; t <= T; t++)
        for (int i = 1; i <= K; i++)
            printf("1000000000000000%c", (i == K) ? '\n' : ' ');
    return 0;
}
