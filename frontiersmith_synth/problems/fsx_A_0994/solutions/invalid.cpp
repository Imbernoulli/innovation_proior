// TIER: invalid
// Deliberately infeasible: claims to build one mirror at site id P+1, which
// is always out of range [1,P]. Must score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int P, M, K;
    if (scanf("%d %d %d", &P, &M, &K) != 3) return 0;
    for (int i = 0; i < P; i++) { long long x, y, q; scanf("%lld %lld %lld", &x, &y, &q); }
    for (int k = 0; k < K; k++) { long long dx, dy, e; scanf("%lld %lld %lld", &dx, &dy, &e); }
    printf("1\n%d\n", P + 1);
    return 0;
}
