// TIER: invalid
// Emits job 1's delivery before its pickup -> precedence violated -> must score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N; long long SX, SY;
    if (scanf("%d %lld %lld", &N, &SX, &SY) != 3) return 0;
    for (int i = 0; i < N; i++) {
        long long a, b, c, d, w;
        scanf("%lld %lld %lld %lld %lld", &a, &b, &c, &d, &w);
    }
    // delivery (-1) then pickup (1): delivery precedes pickup -> infeasible
    printf("2\n-1 1\n");
    return 0;
}
