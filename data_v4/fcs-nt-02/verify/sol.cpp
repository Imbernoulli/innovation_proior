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
