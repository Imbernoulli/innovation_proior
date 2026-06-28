#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
const ll MOD = 1000000007LL;

ll power_mod(ll base, ll exp, ll mod) {
    base %= mod;
    if (base < 0) base += mod;
    ll result = 1;
    while (exp > 0) {
        if (exp & 1) result = (__int128)result * base % mod;
        base = (__int128)base * base % mod;
        exp >>= 1;
    }
    return result;
}

ll inv_mod(ll a, ll mod) { return power_mod(a, mod - 2, mod); }

// Euler's totient phi(m), computed exactly (m fits in 64-bit).
ll phi_exact(ll m) {
    ll result = m;
    for (ll p = 2; p * p <= m; p++) {
        if (m % p == 0) {
            while (m % p == 0) m /= p;
            result -= result / p;
        }
    }
    if (m > 1) result -= result / m;
    return result;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    ll n, k;
    if (!(cin >> n >> k)) return 0;

    ll kk = k % MOD;

    // ---- Rotation part: sum over d=0..n-1 of k^gcd(d,n).
    // Regroup by g = gcd(d,n): for each divisor d of n, exactly phi(n/d)
    // values of the offset have gcd equal to d, contributing k^d each.
    // rotSum = sum_{d | n} phi(n/d) * k^d   (mod p)
    ll rotSum = 0;
    for (ll d = 1; d * d <= n; d++) {
        if (n % d == 0) {
            ll d2 = n / d;
            // divisor d, complement d2
            ll t1 = (phi_exact(n / d) % MOD) * power_mod(kk, d, MOD) % MOD;
            rotSum = (rotSum + t1) % MOD;
            if (d2 != d) {
                ll t2 = (phi_exact(n / d2) % MOD) * power_mod(kk, d2, MOD) % MOD;
                rotSum = (rotSum + t2) % MOD;
            }
        }
    }

    // ---- Reflection part.
    ll reflSum = 0;
    if (n % 2 == 1) {
        // n odd: n axes, each fixes k^((n+1)/2) colorings.
        reflSum = (n % MOD) * power_mod(kk, (n + 1) / 2, MOD) % MOD;
    } else {
        // n even: n/2 axes through two opposite vertices  -> k^(n/2 + 1)
        //         n/2 axes through two opposite edge mids  -> k^(n/2)
        ll half = n / 2;
        ll a = ((half % MOD) * power_mod(kk, half + 1, MOD)) % MOD;
        ll b = ((half % MOD) * power_mod(kk, half, MOD)) % MOD;
        reflSum = (a + b) % MOD;
    }

    // Burnside: distinct = (rotSum + reflSum) / (2n)  over the dihedral group.
    ll total = (rotSum + reflSum) % MOD;
    ll denom = (2 * (n % MOD)) % MOD;
    ll ans = total % MOD * inv_mod(denom, MOD) % MOD;

    cout << ans << "\n";
    return 0;
}
