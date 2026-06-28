#include <bits/stdc++.h>
using namespace std;

// Fast modular exponentiation: base^exp mod m (m prime, fits in 32-bit so products fit in 64-bit).
static long long power_mod(long long base, long long exp, long long m) {
    long long result = 1 % m;
    base %= m;
    if (base < 0) base += m;
    while (exp > 0) {
        if (exp & 1LL) result = (__int128)result * base % m;
        base = (__int128)base * base % m;
        exp >>= 1;
    }
    return result;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int q;
    if (!(cin >> q)) return 0;            // no queries -> nothing to print
    while (q--) {
        long long n, p;
        cin >> n >> p;                    // count length-2n balanced sequences mod prime p

        // Catalan(n) = (2n)! / (n! * (n+1)!) mod p.
        // The guarantee p > 2n means every factor 1..2n is invertible mod p,
        // so the modular division is always well defined.
        long long fact = 1 % p;           // will hold (2n)! mod p
        for (long long i = 1; i <= 2 * n; i++) {
            fact = (__int128)fact * (i % p) % p;
        }

        // Denominator d = n! * (n+1)! mod p, then multiply by its modular inverse.
        long long fn = 1 % p;             // n! mod p
        for (long long i = 1; i <= n; i++) {
            fn = (__int128)fn * (i % p) % p;
        }
        long long fn1 = (__int128)fn * ((n + 1) % p) % p;  // (n+1)! mod p
        long long denom = (__int128)fn * fn1 % p;
        long long inv_denom = power_mod(denom, p - 2, p);  // Fermat inverse, p prime

        long long ans = (__int128)fact * inv_denom % p;
        cout << ans << "\n";
    }
    return 0;
}
