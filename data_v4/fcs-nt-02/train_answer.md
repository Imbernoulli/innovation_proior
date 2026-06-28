**Problem.** Given integers `n`, `r`, `m` with `0 <= r, n <= 10^18` and `1 <= m <= 10^6`, print
`C(n, r) mod m`. The modulus `m` is arbitrary — possibly composite and not squarefree — so the usual
"factorials + Fermat inverses" trick does not apply. If `r > n` the answer is `0`.

**Why the obvious approach is not just slow but undefined.** The prime-modulus recipe
`C = n! * inv(r!) * inv((n-r)!) mod p` (scaled to large `n` by Lucas' theorem) needs each factorial to
be invertible. Modulo a composite `m`, a factorial like `r!` contains every prime dividing `m`, so
`gcd(r!, m) > 1` and `(r!)^{-1} mod m` **does not exist**. Concretely `inv(6) mod 12` is undefined
(`gcd(6,12)=6`), so the inverse-based formula cannot even be evaluated, let alone optimized.

**Key idea — CRT over prime powers, then strip the prime out of every factorial.** Two ideas combine:

1. *Decompose the modulus.* Factor `m = ∏ p^e`. The prime powers are pairwise coprime, so by the
   Chinese Remainder Theorem, `C(n, r) mod m` is determined by `C(n, r) mod p^e` for each factor. This
   reduces the composite problem to a few prime-power problems (at most ~7 distinct primes under `10^6`).

2. *Separate the `p`-part from the unit part.* Modulo `p^e`, write `k! = p^{v_p(k!)} * (k!)_p`, where
   `(k!)_p` (the "`p`-free factorial") is the product over `[1, k]` of the integers **not divisible by
   `p`**. The `p`-free part is coprime to `p`, hence a **unit** mod `p^e`, hence invertible — so the
   ill-defined division becomes legitimate. Then
   `C(n, r) ≡ p^{v_p(C)} * (n!)_p / ((r!)_p (n-r)!)_p) (mod p^e)`, with:
   - **Kummer/Legendre** for the exponent: `v_p(k!) = floor(k/p) + floor(k/p^2) + ...`, and
     `v_p(C(n,r)) = v_p(n!) - v_p(r!) - v_p((n-r)!)`. If this is `>= e`, then `p^e | C(n, r)` and the
     residue mod `p^e` is `0`.
   - **Wilson's generalization** for the periodic block of the `p`-free factorial: the product of all
     units in one block `[1, p^e]` is `-1 mod p^e` for odd `p` (any `e`) and for `p^e ∈ {2, 4}`, and
     `+1` for `p = 2, e >= 3`. So `(x!)_p = blockSign^{floor(x/p^e)} * (partial up to x mod p^e) *
     (floor(x/p)!)_p`, recursing in `O(log_p x)` steps.

Precompute one prefix-product table of size `p^e` per prime power; each factorial evaluation is then
`O(log_p n)`. This is the canonical generalized-Lucas / Granville factorial method.

**Pitfalls.**
1. *Non-existent inverses.* Never invert a full factorial mod composite `m`. Invert only the `p`-free
   parts, which are units; carry the explicit `p^{v_p(C)}` separately.
2. *Block sign.* The Wilson sign flips for `p = 2, e >= 3` (it is `+1`, not `-1`). Reading the sign
   straight from `fact[p^e - 1]` (product of one block) gets it right automatically, no `p=2`
   special-case needed.
3. *CRT merge.* When folding `(a mod p^e)` into the running `(res mod M)`, the inverse must be of
   `M mod p^e`, not `M` (which can exceed `p^e` after the first merge). Passing an unreduced modulus
   corrupts only the **composite** cases — exactly the ones this problem targets — so it can slip past
   all prime-`m` tests. Reduce explicitly: `inv_mod(M % pe, pe)`.
4. *Types.* `n, r` up to `10^18` demand `long long` everywhere; guard products with `__int128`.

**Edge cases.** `m = 1 -> 0` (short-circuited); `r = 0` or `r = n -> 1`; `r > n -> 0`; large prime
powers `2^e/3^e/5^e/7^e` exercise both Wilson signs; full `n = r = 10^18` runs in milliseconds.

**Complexity.** Factoring `m` is `O(sqrt(m))`. Per prime power: `O(p^e)` table build (total `O(m)`) plus
`O(log_p n)` per factorial. Overall `O(m + (#primes) * log n)` — trivially within the limit.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef unsigned long long ull;
typedef __int128 lll;

// ----- 128-bit safe modular multiply (mod fits in <= ~1e6, but base/exponent
//       intermediates use values < p^e <= m <= 1e6, so 64-bit suffices; we still
//       guard products with __int128 to be safe). -----
static inline ll mulmod(ll a, ll b, ll mod) {
    return (ll)((lll)a * b % mod);
}

static ll powmod(ll a, ll e, ll mod) {
    a %= mod; if (a < 0) a += mod;
    ll r = 1 % mod;
    while (e > 0) {
        if (e & 1) r = mulmod(r, a, mod);
        a = mulmod(a, a, mod);
        e >>= 1;
    }
    return r;
}

// Extended Euclid -> modular inverse of a mod m (a must be coprime to m).
static ll inv_mod(ll a, ll m) {
    ll g0 = a % m, g1 = m, x0 = 1, x1 = 0;
    if (g0 < 0) g0 += m;
    while (g1) {
        ll q = g0 / g1;
        ll t = g0 - q * g1; g0 = g1; g1 = t;
        t = x0 - q * x1; x0 = x1; x1 = t;
    }
    // g0 == gcd; assumed 1
    x0 %= m; if (x0 < 0) x0 += m;
    return x0;
}

struct PrimePowerComb {
    ll p;       // prime
    int e;      // exponent
    ll pe;      // p^e
    vector<ll> fact; // fact[i] = product of j in [1..i] with p | j removed, mod pe

    void init(ll p_, int e_) {
        p = p_; e = e_;
        pe = 1; for (int i = 0; i < e; i++) pe *= p;
        fact.assign((size_t)pe + 1, 1);
        fact[0] = 1 % pe;
        for (ll i = 1; i <= pe; i++) {
            if (i % p == 0) fact[i] = fact[i - 1];
            else fact[i] = mulmod(fact[i - 1], i % pe, pe);
        }
    }

    // total power of p dividing x!
    ll legendre(ll x) {
        ll cnt = 0;
        while (x > 0) { x /= p; cnt += x; }
        return cnt;
    }

    // (x!)_p  : x! with all factors of p stripped, taken mod pe.
    ll factmod(ll x) {
        ll res = 1 % pe;
        while (x > 0) {
            // number of complete blocks of length pe in [1..x]
            ll blocks = x / pe;
            // product of one full block == fact[pe-1] (== product of all units mod pe);
            // by Wilson's generalization this is -1 mod pe for odd p (any e) and for
            // p^e in {2,4}; +1 for p=2,e>=3. We just read it from the table directly.
            ll blockProd = fact[pe - 1];
            res = mulmod(res, powmod(blockProd, blocks, pe), pe);
            res = mulmod(res, fact[x % pe], pe);
            x /= p; // recurse on floor(x/p): the multiples of p contribute (x/p)! again
        }
        return res;
    }

    // nCr mod pe
    ll comb(ll n, ll r) {
        if (r < 0 || r > n) return 0;
        ll powp = legendre(n) - legendre(r) - legendre(n - r); // power of p in nCr
        if (powp >= e) return 0; // divisible by pe
        ll num = factmod(n);
        ll den = mulmod(factmod(r), factmod(n - r), pe);
        ll res = mulmod(num, inv_mod(den, pe), pe);
        // multiply back the surviving p^powp
        for (ll i = 0; i < powp; i++) res = mulmod(res, p % pe, pe);
        return res % pe;
    }
};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    ll n, r, m;
    if (!(cin >> n >> r >> m)) return 0;

    if (m == 1) { cout << 0 << "\n"; return 0; }

    // edge: r out of range -> 0 (mod m). Handle generally inside comb too.
    // Factor m into prime powers.
    vector<pair<ll,int>> pf;
    ll mm = m;
    for (ll d = 2; d * d <= mm; d++) {
        if (mm % d == 0) {
            int cnt = 0;
            while (mm % d == 0) { mm /= d; cnt++; }
            pf.push_back({d, cnt});
        }
    }
    if (mm > 1) pf.push_back({mm, 1});

    // CRT accumulation: result `res` mod `mod`.
    ll res = 0, mod = 1;
    for (auto &pr : pf) {
        PrimePowerComb ppc;
        ppc.init(pr.first, pr.second);
        ll a = ppc.comb(n, r); // value mod pe
        ll pe = ppc.pe;
        // CRT merge (res, mod) with (a, pe); mod and pe are coprime.
        // Find t with res + mod*t == a (mod pe):  t = (a-res) * mod^{-1} (mod pe).
        ll inv = inv_mod(mod % pe, pe);         // mod is coprime to pe
        ll diff = ((a - res) % pe + pe) % pe;
        ll t = mulmod(diff, inv, pe);
        res = res + mod * t;
        mod = mod * pe;
        res %= mod;
        if (res < 0) res += mod;
    }

    cout << (res % m) << "\n";
    return 0;
}
```
