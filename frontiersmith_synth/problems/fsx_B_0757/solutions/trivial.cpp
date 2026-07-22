// TIER: trivial
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Reproduces exactly the checker's internal baseline: a single 1-wide pillar of
// height h0=min(M,H) at the middle column. This is the "do nothing clever"
// reference construction, so it lands exactly at ratio 0.1.

int main() {
    ll W, H, S, M, Fx, Fy;
    cin >> W >> H >> S >> M >> Fx >> Fy;
    ll x0 = (W + 1) / 2;
    ll h0 = min(M, H);
    if (h0 < 1) h0 = 1;
    printf("%lld\n", h0);
    for (ll y = 1; y <= h0; y++) printf("%lld %lld\n", x0, y);
    return 0;
}
