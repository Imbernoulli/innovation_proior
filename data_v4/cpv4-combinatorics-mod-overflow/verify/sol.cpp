#include <bits/stdc++.h>
using namespace std;

typedef long long ll;

ll MOD;

ll mul(ll a, ll b) {
    return (a % MOD) * (b % MOD) % MOD;   // 64-bit product; int would overflow here
}

ll power(ll base, ll e) {
    ll r = 1 % MOD;
    base %= MOD;
    if (base < 0) base += MOD;
    while (e > 0) {
        if (e & 1) r = mul(r, base);
        base = mul(base, base);
        e >>= 1;
    }
    return r;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int q;
    if (!(cin >> q >> MOD)) return 0;

    // Maximum coordinate value across the whole input bounds factorial table size.
    // We pre-read all queries to know the max factorial index needed.
    // R + U for a leg can be up to (X + Y) <= 2*10^6.
    struct Query { ll cx, cy, ex, ey; };
    vector<Query> qs(q);
    long long maxN = 0;
    for (int i = 0; i < q; i++) {
        cin >> qs[i].cx >> qs[i].cy >> qs[i].ex >> qs[i].ey;
        maxN = max(maxN, qs[i].cx + qs[i].cy);
        maxN = max(maxN, (qs[i].ex - qs[i].cx) + (qs[i].ey - qs[i].cy));
    }

    // Factorials and inverse factorials mod MOD (MOD is prime, MOD > maxN).
    vector<ll> fact(maxN + 1), inv_fact(maxN + 1);
    fact[0] = 1 % MOD;
    for (ll i = 1; i <= maxN; i++) fact[i] = mul(fact[i - 1], i % MOD);
    inv_fact[maxN] = power(fact[maxN], MOD - 2);
    for (ll i = maxN; i >= 1; i--) inv_fact[i - 1] = mul(inv_fact[i], i % MOD);

    auto C = [&](ll n, ll k) -> ll {
        if (k < 0 || k > n) return 0;
        return mul(fact[n], mul(inv_fact[k], inv_fact[n - k]));
    };

    for (int i = 0; i < q; i++) {
        ll cx = qs[i].cx, cy = qs[i].cy, ex = qs[i].ex, ey = qs[i].ey;
        // Paths (0,0)->(cx,cy): choose which of the (cx+cy) steps go right (cx of them).
        ll leg1 = C(cx + cy, cx);
        // Paths (cx,cy)->(ex,ey): need ex-cx rights and ey-cy ups.
        ll dx = ex - cx, dy = ey - cy;
        ll leg2 = C(dx + dy, dx);
        cout << mul(leg1, leg2) << "\n";   // product also needs 64-bit before reduction
    }
    return 0;
}
