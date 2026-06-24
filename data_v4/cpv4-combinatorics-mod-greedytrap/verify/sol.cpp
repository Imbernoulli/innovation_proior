#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

ll MOD;

// We precompute factorials and inverse factorials modulo a PRIME p, up to the
// largest a+b appearing in any query, then answer each query in O(1) via the
// reflection (Andre) formula:
//   safe(a,b,m) = C(a+b, a) - C(a+b, b-m-1)   (the subtracted term is 0 when its
//   lower index is outside [0, a+b]), and 0 entirely when a-b < -m.

vector<ll> fct, inv_fct;

ll pw(ll base, ll e, ll mod) {
    base %= mod; if (base < 0) base += mod;
    ll r = 1 % mod;
    while (e > 0) {
        if (e & 1) r = (__int128)r * base % mod;
        base = (__int128)base * base % mod;
        e >>= 1;
    }
    return r;
}

void build(ll maxn) {
    fct.assign(maxn + 1, 0);
    inv_fct.assign(maxn + 1, 0);
    fct[0] = 1 % MOD;
    for (ll i = 1; i <= maxn; i++) fct[i] = (__int128)fct[i - 1] * (i % MOD) % MOD;
    inv_fct[maxn] = pw(fct[maxn], MOD - 2, MOD);   // Fermat: p prime
    for (ll i = maxn; i >= 1; i--) inv_fct[i - 1] = (__int128)inv_fct[i] * (i % MOD) % MOD;
}

// C(n, k) mod p, with the standard out-of-range = 0 convention.
ll C(ll n, ll k) {
    if (k < 0 || k > n) return 0;
    return (__int128)fct[n] * inv_fct[k] % MOD * inv_fct[n - k] % MOD;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int q;
    if (!(cin >> q >> MOD)) return 0;

    struct Query { ll a, b, m; };
    vector<Query> qs(q);
    ll maxn = 0;
    for (int i = 0; i < q; i++) {
        cin >> qs[i].a >> qs[i].b >> qs[i].m;
        maxn = max(maxn, qs[i].a + qs[i].b);
    }

    build(maxn);

    for (int i = 0; i < q; i++) {
        ll a = qs[i].a, b = qs[i].b, m = qs[i].m;
        ll ans;
        if (a - b < -m) {
            ans = 0;                              // cannot even finish within margin
        } else {
            ll total = C(a + b, a);               // all interleavings
            ll bad = C(a + b, b - m - 1);         // reflected = orderings that cross the barrier
            ans = (total - bad) % MOD;
            if (ans < 0) ans += MOD;
        }
        cout << ans << "\n";
    }
    return 0;
}
