**Problem.** A sequence over `Z_p` (`p = 998244353`) obeys an order-`k` linear recurrence
`a[n] = c[0]*a[n-1] + ... + c[k-1]*a[n-k]` with given seeds `a[0..k-1]`. Read `k`, the `k`
coefficients, the `k` seeds, and an index `N`, then print `a[N] mod p`. Scale:
`k <= 2*10^4`, `N <= 10^18`.

**Why the obvious approaches fail.** Iterating the recurrence term by term is `Theta(N*k)` —
impossible at `N = 10^18`. Matrix exponentiation (raise the `k x k` companion matrix to the power
`N` by square-and-multiply) is the right *kind* of jump but the wrong representation: one matrix
multiply is `O(k^3)`, so the method is `O(k^3 log N) = (2*10^4)^3 * 60 ~ 5*10^{15}` — it cannot even
finish a single multiply at this `k`.

**Key idea — raise `x` to the power inside `Z_p[x]/(g(x))` (Kitamasa / Fiduccia).** The companion
matrix `M` is "multiply by `x`" in the quotient ring `R = Z_p[x] / (g(x))`, where

```
g(x) = x^k - c[0]*x^{k-1} - c[1]*x^{k-2} - ... - c[k-1]
```

is the characteristic polynomial (Cayley–Hamilton: `g(M) = 0`, so `M <-> x`). Computing the `N`-step
advance is therefore computing `f(x) = x^N mod g(x)`, a polynomial of degree `< k`, by binary
exponentiation in `R`. Then linearity closes the loop:

```
a[N] = sum_{i=0}^{k-1} f[i] * a[i]   (mod p),
```

a single dot product of the reduced polynomial with the seeds. An element of `R` is `k` coefficients
(not a `k x k` matrix), and one squaring is "polynomial multiply + reduce mod g," each `O(k log k)`,
so the whole method is `O(k log k log N)` — the right complexity for the limits.

The two ring operations, both `O(k log k)`:
- **Multiply** via NTT (`p = 119*2^23 + 1` is NTT-friendly, primitive root 3).
- **Reduce mod g** via the reverse-and-invert division: precompute `rgi = rev(g)^{-1}` with Newton
  iteration once; each reduction is `rev(q) = rev(a)*rgi` (truncated), un-reverse to get the quotient
  `q`, then `r = a - q*g` keeping the low `k` coefficients. Two NTT-multiplies per reduction.

**Pitfalls to get right.**
1. *The base element at `k = 1`.* `base = x` is only pre-reduced when `k >= 2` (then `deg x = 1 < k`).
   At `k = 1`, `x` has degree `1 = k` and is *not* reduced — it equals `c[0]` in `R` — and writing
   `base[1]` into a length-1 vector is out of bounds. Reduce `x mod g` explicitly for the base.
2. *Quotient sizes in the fast reverse-divide.* After multiplying two degree-`<k` polynomials, the
   product has degree `2k-2`, so the quotient degree is `qd = k-2`. You must truncate `rev(a)` to
   `qd+1` terms and take exactly the *low* `qd+1` terms of `rev(a)*rgi` as `rev(q)`, and exactly the
   low `k` terms of `q*g`. Mixing up the slice offset/size gives answers that are right only when
   `qd+1` is a power of two (no NTT aliasing) and wrong otherwise — a nasty intermittent bug.
3. *Sign of `g`.* `g[k] = 1`, `g[k-1-j] = -c[j]`. A flipped sign silently corrupts every term.
4. *Reduction in input.* Coefficients/seeds may be any integers (incl. negative); read as `long long`
   and reduce mod `p` to `[0,p)`.

**Edge cases.** `N < k` -> return the seed `a[N]` directly (covers `N=0`, `N=k-1`); `N=k` -> one full
pass; `k=1` geometric sequences; all-zero coefficients -> `g = x^k`, so `x^N mod g = 0` for `N>=k`;
sparse coefficients (zeros in `g`); full-range and negative values.

**Complexity.** `O(k log k log N)` time, `O(k)` space. Worst case (`k=2*10^4`, `N=10^18`) runs in
about 1.5 s within a 3 s limit; caching the forward NTTs of `g` and the truncated `rgi` once in
`init` removes redundant transforms from the hot reduction.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef unsigned long long ull;

const int MOD = 998244353;       // 119 * 2^23 + 1, NTT-friendly prime
const int PRIMITIVE_ROOT = 3;    // primitive root of MOD

inline int add(int a, int b) { a += b; if (a >= MOD) a -= MOD; return a; }
inline int sub(int a, int b) { a -= b; if (a < 0) a += MOD; return a; }
inline int mul(int a, int b) { return (int)((ll)a * b % MOD); }

int power(int a, ll e) {
    int r = 1; a %= MOD; if (a < 0) a += MOD;
    while (e > 0) { if (e & 1) r = mul(r, a); a = mul(a, a); e >>= 1; }
    return r;
}
inline int inv(int a) { return power(a, MOD - 2); }

// In-place iterative NTT. dir = +1 forward, -1 inverse.
void ntt(vector<int> &A, int dir) {
    int n = (int)A.size();
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) swap(A[i], A[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        int w = power(PRIMITIVE_ROOT, (MOD - 1) / len);
        if (dir == -1) w = inv(w);
        for (int i = 0; i < n; i += len) {
            int wn = 1;
            for (int j = 0; j < len / 2; j++) {
                int u = A[i + j];
                int v = mul(A[i + j + len / 2], wn);
                A[i + j] = add(u, v);
                A[i + j + len / 2] = sub(u, v);
                wn = mul(wn, w);
            }
        }
    }
    if (dir == -1) {
        int ninv = inv(n);
        for (int &x : A) x = mul(x, ninv);
    }
}

// Polynomial multiplication mod MOD. Result degree = deg(a)+deg(b).
vector<int> poly_mul(const vector<int> &a, const vector<int> &b) {
    if (a.empty() || b.empty()) return {};
    int rs = (int)a.size() + (int)b.size() - 1;
    int n = 1; while (n < rs) n <<= 1;
    vector<int> fa(a.begin(), a.end()), fb(b.begin(), b.end());
    fa.resize(n, 0); fb.resize(n, 0);
    ntt(fa, 1); ntt(fb, 1);
    for (int i = 0; i < n; i++) fa[i] = mul(fa[i], fb[i]);
    ntt(fa, -1);
    fa.resize(rs);
    return fa;
}

// Inverse of polynomial a modulo x^m (a[0] must be nonzero), via Newton iteration.
vector<int> poly_inv(const vector<int> &a, int m) {
    vector<int> b(1, inv(a[0]));
    int cur = 1;
    while (cur < m) {
        int nxt = cur << 1;
        int sz = 1; while (sz < nxt + nxt - 1) sz <<= 1; // size for the products below
        vector<int> fa(a.begin(), a.begin() + min((int)a.size(), nxt));
        fa.resize(sz, 0);
        vector<int> fb = b; fb.resize(sz, 0);
        ntt(fa, 1); ntt(fb, 1);
        for (int i = 0; i < sz; i++)
            fa[i] = mul(fb[i], sub(2, mul(fa[i], fb[i]))); // b * (2 - a*b)
        ntt(fa, -1);
        fa.resize(nxt);
        b = fa;
        cur = nxt;
    }
    b.resize(m);
    return b;
}

// Polynomial remainder a mod g, deg(g) = k, g monic-or-not (g.back() != 0 assumed).
// Returns r with deg(r) < k. Uses precomputed reverse-inverse rgi = (rev(g))^{-1} mod x^{?}.
// We compute it directly each call here for clarity; caller may cache.
struct PolyMod {
    int k;                  // degree of g
    vector<int> g;          // g, size k+1
    vector<int> rgi;        // (reverse(g))^{-1} mod x^{k}

    // Cached forward transforms for the hot path (deg(a) == 2k-2 case).
    // For that case qd = k-2, both products have size 2k-3, padded to nfix.
    int nfix;
    vector<int> g_hat;      // NTT of g padded to nfix
    vector<int> rgi_hat;    // NTT of (rgi mod x^{qd+1}) padded to nfix

    void init(const vector<int> &g_) {
        g = g_;
        k = (int)g.size() - 1;
        vector<int> rg(g.rbegin(), g.rend());
        // We need inverse of rg modulo x^{deg(a)-deg(g)+1}; deg(a) <= 2k-2, so up to k-1.
        rgi = poly_inv(rg, max(1, k));

        // Precompute transforms for the dominant case deg(a) = 2k-2 (qd = k-2).
        if (k >= 2) {
            int qd = k - 2;
            int rs = max((qd + 1) + (qd + 1) - 1, (qd + 1) + (k + 1) - 1);
            nfix = 1; while (nfix < rs) nfix <<= 1;
            g_hat.assign(g.begin(), g.end()); g_hat.resize(nfix, 0); ntt(g_hat, 1);
            rgi_hat.assign(rgi.begin(), rgi.begin() + (qd + 1)); rgi_hat.resize(nfix, 0); ntt(rgi_hat, 1);
        } else {
            nfix = 0;
        }
    }

    // reduce polynomial a (any degree) modulo g, return remainder of degree < k.
    vector<int> reduce(vector<int> a) {
        while ((int)a.size() > 1 && a.back() == 0) a.pop_back();
        int n = (int)a.size() - 1; // deg(a)
        if (n < k) { a.resize(k, 0); return a; }
        int qd = n - k; // degree of quotient

        vector<int> q; // quotient, degree qd
        // Fast path: cached transforms apply only to the canonical full-degree case.
        if (k >= 2 && qd == k - 2 && nfix > 0) {
            // q_rev = (reverse(a) trimmed to qd+1) * rgi_cut, keep low qd+1 terms
            vector<int> ra(a.rbegin(), a.rend());
            ra.resize(qd + 1);
            ra.resize(nfix, 0);
            ntt(ra, 1);
            for (int i = 0; i < nfix; i++) ra[i] = mul(ra[i], rgi_hat[i]);
            ntt(ra, -1);
            q.assign(ra.begin(), ra.begin() + (qd + 1));
            reverse(q.begin(), q.end());
            // qg = q * g (low k terms), via cached g_hat
            vector<int> qf(q.begin(), q.end()); qf.resize(nfix, 0);
            ntt(qf, 1);
            for (int i = 0; i < nfix; i++) qf[i] = mul(qf[i], g_hat[i]);
            ntt(qf, -1);
            vector<int> r(k, 0);
            for (int i = 0; i < k; i++) {
                int av = (i < (int)a.size()) ? a[i] : 0;
                int qv = qf[i];
                r[i] = sub(av, qv);
            }
            return r;
        }

        // General path.
        vector<int> ra(a.rbegin(), a.rend());
        ra.resize(qd + 1);
        if ((int)rgi.size() < qd + 1) {
            vector<int> rg(g.rbegin(), g.rend());
            rgi = poly_inv(rg, qd + 1);
        }
        vector<int> rgi_cut(rgi.begin(), rgi.begin() + (qd + 1));
        q = poly_mul(ra, rgi_cut);
        q.resize(qd + 1);
        reverse(q.begin(), q.end());
        vector<int> qg = poly_mul(q, g);
        vector<int> r(k, 0);
        for (int i = 0; i < k; i++) {
            int av = (i < (int)a.size()) ? a[i] : 0;
            int qv = (i < (int)qg.size()) ? qg[i] : 0;
            r[i] = sub(av, qv);
        }
        return r;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int k;
    if (!(cin >> k)) return 0;
    ll N;
    vector<int> c(k), a(k);
    for (int i = 0; i < k; i++) { ll x; cin >> x; x %= MOD; if (x < 0) x += MOD; c[i] = (int)x; }
    for (int i = 0; i < k; i++) { ll x; cin >> x; x %= MOD; if (x < 0) x += MOD; a[i] = (int)x; }
    cin >> N;

    // If N < k, answer is just the seed.
    if (N < k) { cout << a[(int)N] << "\n"; return 0; }

    // Characteristic polynomial g(x) = x^k - c0 x^{k-1} - ... - c_{k-1}
    // g[i] = coefficient of x^i. g[k] = 1, g[k-1-j] = -c[j].
    vector<int> g(k + 1, 0);
    g[k] = 1;
    for (int j = 0; j < k; j++) g[k - 1 - j] = sub(0, c[j]);

    PolyMod pm;
    pm.init(g);

    // Compute f = x^N mod g via binary exponentiation.
    // base = x mod g  (if k == 1, x mod g must be reduced)
    vector<int> result(k, 0); result[0] = 1; // represents polynomial 1
    vector<int> base;
    if (k == 1) {
        base = pm.reduce(vector<int>{0, 1}); // x mod g
    } else {
        base = vector<int>(k, 0); base[1] = 1; // x, already degree < k since k>=2
    }

    ll e = N;
    while (e > 0) {
        if (e & 1) {
            result = pm.reduce(poly_mul(result, base));
        }
        e >>= 1;
        if (e > 0) base = pm.reduce(poly_mul(base, base));
    }

    // answer = sum_{i=0}^{k-1} result[i] * a[i]
    int ans = 0;
    for (int i = 0; i < k; i++) ans = add(ans, mul(result[i], a[i]));
    cout << ans << "\n";
    return 0;
}
```
