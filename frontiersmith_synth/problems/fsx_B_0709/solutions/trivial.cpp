// TIER: trivial
// Fixed uniform grid over [0,L]x[0,L], sized only from K -- ignores weights,
// resilience targets and jammer positions entirely. This is exactly the
// checker's internal baseline B, so it scores ratio ~0.1 by construction.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    int N, K, M, L, r, R;
    if (scanf("%d %d %d %d %d %d", &N, &K, &M, &L, &r, &R) != 6) return 0;
    int s = (int)ceil(sqrt((double)K));
    if (s < 1) s = 1;
    for (int idx = 0; idx < K; idx++){
        int row = idx / s, col = idx % s;
        ll x = (ll)(2 * col + 1) * L / (2LL * s);
        ll y = (ll)(2 * row + 1) * L / (2LL * s);
        printf("%lld %lld\n", x, y);
    }
    return 0;
}
