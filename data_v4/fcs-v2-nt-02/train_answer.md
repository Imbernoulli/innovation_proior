**Problem.** Count distinct bracelets: `n` beads in a circle, `k` colors per bead, where two
colorings are the same bracelet if one maps to the other under the dihedral group `D_n` (any of the
`n` rotations or `n` reflections). Output the count modulo `p = 1000000007`. Constraints:
`1 <= n <= 10^9`, `1 <= k <= 10^9`.

**Why the obvious approach is too slow.** Burnside's lemma gives
`answer = (1/(2n)) * (sum over rotations of Fix + sum over reflections of Fix)`. A rotation by `d`
fixes `k^{gcd(d,n)}` colorings, so the rotation sum is `sum_{d=0}^{n-1} k^{gcd(d,n)}`. Written this
way it is `O(n)` modular exponentiations — `10^9` of them at the top constraint, which cannot finish
in the time limit. The naive Burnside sum is exactly what the constraints rule out.

**Key idea (the insight) — regroup the rotation sum by the gcd into a divisor sum.** The exponent
`gcd(d,n)` only ever equals a **divisor** of `n`. The number of offsets `d in [0,n-1]` with
`gcd(d,n) = g` (for `g | n`) is exactly Euler's totient `phi(n/g)` (write `d = g*t`, then
`gcd(t, n/g) = 1`). Therefore

```
rotation part = sum_{g | n} phi(n/g) * k^{g}   (equivalently  sum_{g|n} phi(g) * k^{n/g}).
```

This turns `n` terms into one term per divisor of `n` — at most a few thousand. Enumerate divisors in
`O(sqrt n)` (pair each `d <= sqrt n` with `n/d`), and compute each `phi(m)` for a single
`m <= 10^9` by trial division in `O(sqrt m)` (no sieve). That kills the `O(n)` wall.

**Reflections (an `O(1)` parity-dependent term).** Their cycle count depends on parity and axis type:
- `n` odd: `n` axes, each through one bead and one opposite gap -> `(n+1)/2` cycles -> `n * k^{(n+1)/2}`.
- `n` even: `n/2` axes through two opposite beads -> `n/2 + 1` cycles -> `k^{n/2+1}`; and `n/2` axes
  through two opposite gaps -> `n/2` cycles -> `k^{n/2}`. Total `(n/2)(k^{n/2+1} + k^{n/2})`.

**Final formula.** `answer = (rotation part + reflection part) * inverse(2n)  (mod p)`.

**Pitfalls to get right.**
1. *Perfect-square double-count.* When enumerating divisors, the pair `(d, n/d)` collapses for square
   `n` at `d = sqrt n` (and at `n = 1`). Add the complement term only when `n/d != d`, or you double
   it. (Tracing `n = 1` returning `14` instead of `7` exposes exactly this.)
2. *Modular division must be well-defined.* Dividing by `2n` is multiplication by `inverse(2n) mod p`,
   which needs `gcd(2n, p) = 1`. Choosing `p = 10^9 + 7 > n` guarantees `p` is odd and `p > n >= n`,
   so `p` divides neither `2` nor `n` — the inverse always exists. A prime `< 10^9` could be divided
   by some valid `n` and silently break.
3. *Reduce `k` first, use 64-bit.* `k` up to `10^9` is reduced mod `p` once before exponentiation;
   `2*(n mod p)` can exceed 32 bits, so keep everything in `long long` and multiply through
   `__int128` inside `power_mod`.

**Edge cases.** `k = 1` -> exactly `1` bracelet for any `n` (everything collapses to `2n/2n`).
`n = 1` -> `k` bracelets. `n = 2, k = 2` -> `3`. Prime `n` (only two divisors) and highly composite
`n` (~1344 divisors near `10^9`) both run in well under a millisecond. Sanity anchors: `(4,2)->6`,
`(6,3)->92`, `(10,4)->53764`.

**Complexity.** `O(sqrt n)` for divisor enumeration, with one `phi` (each `O(sqrt(n/d))`) and one
`power_mod` (`O(log n)`) per divisor; reflections are `O(log n)`. Total comfortably within 1 second
for `n` up to `10^9`. `O(1)` extra memory.

**Code.**

```cpp
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
```
