// TIER: invalid
// Deliberately infeasible: activates an out-of-range sensor index -> checker rejects -> 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, M;
    if (scanf("%d %d", &N, &M) != 2) return 0;
    for (int i = 0; i < N; i++) { long long a, b; scanf("%lld %lld", &a, &b); }
    for (int j = 0; j < M; j++) { long long a, b, c, d; scanf("%lld %lld %lld %lld", &a, &b, &c, &d); }
    // K=2, second index is far out of range -> readInt(1,M) fails -> WA -> score 0
    printf("2\n1 100000000\n");
    return 0;
}
