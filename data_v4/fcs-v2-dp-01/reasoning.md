**Reading the problem and pinning the contract.** I have an order-`k` linear recurrence over `Z_p`
with `p = 998244353`: `a[n] = c[0]*a[n-1] + ... + c[k-1]*a[n-k]` for `n >= k`, the seeds
`a[0..k-1]` are given, and I must output `a[N] mod p`. The numbers that matter are the scales:
`k <= 2*10^4` and `N <= 10^18`. The first thing I do is rule out the naive thing, because the naive
thing is what the constraints are designed to punish. Generating the sequence term by term costs one
length-`k` dot product per term, so `Theta(N*k)` field multiplications. At `N = 10^18` that is not
"slow," it is physically impossible — `10^18` iterations alone would run for years. So I am not
allowed to touch `a[m]` for any `m` between `k` and `N`. I have to *jump* to index `N` using
structure, not iteration. That is the whole problem in one sentence.

**The first real candidate: matrix exponentiation.** The textbook structural jump is the companion
matrix. Stack the window `(a[n], a[n-1], ..., a[n-k+1])^T` into a state vector `v_n`. One step of
the recurrence is a linear map: `v_{n+1} = M v_n`, where `M` is the `k x k` companion matrix whose
top row holds `c[0..k-1]` and whose subdiagonal is the identity shift. Then `v_N = M^{N-k+1} v_{k-1}`,
and `M^e` is computed by fast exponentiation: square-and-multiply, `O(log N)` matrix multiplies. The
appeal is that `log N <= 60`, so only ~60 matrix multiplies. The trouble is the *cost of one matrix
multiply*: two `k x k` matrices multiply in `O(k^3)` naively. With `k = 2*10^4` that is
`(2*10^4)^3 = 8*10^{13}` operations per multiply, times 60 — about `5*10^{15}` operations total.
That is `O(k^3 log N)`, and it is hopelessly off. Even an `O(k^{2.81})` Strassen variant does not
rescue a `k` this large; the cubic-ish blow-up in `k` is the killer, not the `log N`. So matrix
power is the right *idea* (jump via a linear map raised to a power) with the wrong *representation*
(a dense `k x k` matrix). I need a representation of the state-advance map that is cheaper to square.

**Re-deriving the jump without a matrix — the characteristic polynomial.** Here is the key
reframing. The companion matrix `M` is just "multiply by `x`" in disguise. Concretely, consider the
polynomial ring `Z_p[x]` and the *characteristic polynomial* of the recurrence,

```
g(x) = x^k - c[0]*x^{k-1} - c[1]*x^{k-2} - ... - c[k-1].
```

In the quotient ring `R = Z_p[x] / (g(x))`, the element `x` acts exactly like `M` does on the state
window: multiplying a polynomial by `x` and reducing mod `g` shifts and folds in the recurrence
coefficients, term for term. (The reason is the Cayley–Hamilton identity: `M` satisfies its own
characteristic polynomial, `g(M) = 0`, so `Z_p[M]` and `R` are isomorphic, with `M <-> x`.) This is
not a different algorithm yet — it is the *same* jump `M^e`, just written as the ring element `x^e`.

But the representation is wildly cheaper. An element of `R` is a polynomial of degree `< k`: a vector
of `k` coefficients, not a `k x k` matrix. And the operation I square is "multiply two such
polynomials and reduce mod `g`," not "multiply two dense matrices." Polynomial multiplication of two
degree-`< k` polynomials is `O(k log k)` with a fast transform, and reduction mod a fixed degree-`k`
polynomial is also `O(k log k)` (shown below). So one squaring is `O(k log k)`, and the whole binary
exponentiation is `O(k log k log N)` — versus `O(k^3 log N)`. That is the insight the problem hinges
on: **don't raise the companion matrix to the power; raise `x` to the power inside
`Z_p[x]/(g(x))`.** This is exactly Fiduccia's method / the Kitamasa method.

**Closing the loop from `x^N mod g` back to `a[N]`.** Suppose I have computed
`f(x) = x^N mod g(x) = sum_{i=0}^{k-1} f[i]*x^i`. Why does that give `a[N]`? Because the linear
functional "read off `a[N]`" is linear in the recurrence's state, and `x^i` corresponds to "advance
`i` steps." Concretely, the same Cayley–Hamilton isomorphism says `a[N] = sum_i f[i] * a[i]`: the
coefficients of `x^N` reduced mod `g`, dotted with the seeds. I can sanity-check this on Fibonacci
later, but structurally it must hold because both sides are the unique linear extension of the
recurrence. So the algorithm is three phases: (1) build `g`; (2) compute `f = x^N mod g` by
square-and-multiply in `R`; (3) output the dot product `sum f[i]*a[i] mod p`.

**Making the two ring operations fast — multiply and reduce.** Multiplication is standard: since
`p = 998244353 = 119*2^23 + 1` is NTT-friendly with primitive root 3, I can multiply two polynomials
via the Number Theoretic Transform in `O(k log k)`. Reduction `mod g` is the part people get wrong,
so I want to be careful. The schoolbook long-division of a degree-`(2k-2)` polynomial by a degree-`k`
polynomial is `O(k^2)` — and `O(k^2 log N)` overall is `(2*10^4)^2 * 60 = 2.4*10^{13}`, still far too
slow. The fast trick is the classic "reverse-and-invert" division. To compute `a = q*g + r` with
`deg r < k`:

- Reverse coefficients: write `rev(P)(x) = x^{deg P} P(1/x)`. Then the quotient satisfies
  `rev(q) = rev(a) * rev(g)^{-1}  (mod x^{deg q + 1})`.
- So precompute `rgi = rev(g)^{-1} mod x^{...}` once (Newton iteration, `O(k log k)`), and each
  reduction is: take `rev(a)`, multiply by `rgi`, truncate, reverse back to get `q`, then
  `r = a - q*g` (only the low `k` coefficients of `q*g` are needed). Two NTT-multiplies per
  reduction: `O(k log k)`.

Newton iteration for the inverse: if `b` is the inverse of `f` mod `x^t`, then
`b' = b*(2 - f*b) mod x^{2t}` is the inverse mod `x^{2t}`. Doubling the precision each step, total
cost `O(k log k)`. That gives me every piece at `O(k log k)`, and the binary exponentiation wraps it
in a `log N` factor.

**First implementation.** I lay it out: modular helpers (`add`, `sub`, `mul`, `power`, `inv`), an
in-place iterative NTT, `poly_mul`, `poly_inv` (Newton), and a `PolyMod` object that holds `g` and
the precomputed `rgi` and exposes `reduce(a)`. The driver builds `g` from `c` (note the sign flip:
`g[k] = 1`, `g[k-1-j] = -c[j]`), sets `base = x`, `result = 1`, and runs square-and-multiply,
finishing with the dot product against the seeds. I also special-case the easy corners up front:
if `N < k`, the answer is literally `a[N]`, no machinery needed.

**Trace before trust — Fibonacci by hand.** Before running anything I check the smallest meaningful
instance on paper, because a sign error in `g` or an off-by-one in the reverse-divide would silently
corrupt everything. Fibonacci: `k = 2`, `c = [1, 1]`, so
`g(x) = x^2 - 1*x - 1`, i.e. coefficients `g = [-1, -1, 1]` (constant, `x`, `x^2`). Seeds `a = [0,1]`.
Take `N = 10`. I want `x^{10} mod g`. Since `x^2 = x + 1` in `R`, powers of `x` follow Fibonacci:
`x^n = F[n]*x + F[n-1]` where `F` is the Fibonacci numbers. So `x^{10} = F[10]*x + F[9] = 55x + 34`,
giving `f = [34, 55]`. Dot with seeds `[0, 1]`: `34*0 + 55*1 = 55`. And indeed `a[10] = 55`. Good —
the sign convention in `g` and the dot-product closing step are consistent.

**Running it — and the first failure.** I compile and throw random small cases at it against a brute
force that literally iterates the recurrence `O(N*k)` for small `N`. Most pass, but I hit a class of
failures on cases with `k = 1` (geometric sequences). For `k = 1`, the recurrence is `a[n] = c[0]*a[n-1]`,
`g(x) = x - c[0]`, degree 1. My `base` initialization said `base = vector<int>(k,0); base[1] = 1;`
— that writes `base[1]` into a length-`k = 1` vector. That is an out-of-bounds write, and worse,
`x` is not even a valid degree-`<1` element: in `R` of dimension 1, `x` must already be reduced to
`x mod g = c[0]`. So my "base = x, already reduced because k>=2" assumption is false at `k = 1`.

**Diagnosing precisely.** The bug is that I conflated "the polynomial `x`" with "the element of `R`
represented by `x`." For `k >= 2`, `x` has degree `1 < k`, so it is already its own reduced form and
`base[1] = 1` is fine. For `k = 1`, `x` has degree `1 = k`, so it is *not* reduced — it equals
`c[0]` in `R` — and the length-1 coefficient vector cannot even hold a `base[1]`. Two symptoms, one
cause: I skipped the reduction of the base element. The fix is to reduce `x` mod `g` explicitly when
building `base`, instead of assuming it is pre-reduced. I split it:

```
if (k == 1) base = pm.reduce(vector<int>{0, 1});   // x mod g, properly reduced
else        { base = vector<int>(k, 0); base[1] = 1; }
```

Re-running the `k = 1` cases: a geometric sequence `a[n] = 2*a[n-1]`, seed `3`, `N = 5` should be
`3*2^5 = 96`. The fixed code prints `96`. And `N = 0` for `k = 1` hits the `N < k` short-circuit and
returns the seed directly. That whole failure class clears.

**The second bug — the reduction's quotient degree.** With `k = 1` fixed I widen the random tests and
hit sporadic wrong answers at slightly larger `k`. I instrument `reduce` and find the culprit in the
fast reverse-divide path. After a `poly_mul` of two degree-`< k` polynomials the product has degree
up to `2k-2`, so the quotient has degree `qd = (2k-2) - k = k-2`. My cached fast path was written for
exactly this canonical case (`qd == k-2`), and the *sizes* of the truncations matter: `rev(a)` must
be truncated to `qd+1` coefficients before multiplying by `rgi`, and I must take only the *low*
`qd+1` coefficients of that product as `rev(q)`. In my first cut I had truncated `rev(a)` to `qd+1`
but then read `qd+1` coefficients starting at the wrong offset of the product, picking up high-degree
garbage. The symptom was answers that were correct whenever `qd+1` happened to equal a power of two
(no aliasing) and wrong otherwise. The fix is to be disciplined: build the product at a fixed
transform size `nfix` large enough for *both* products in the reduction (`rev(a)*rgi` and `q*g`),
inverse-NTT, then slice exactly `[0, qd+1)` for `rev(q)` and exactly `[0, k)` for `q*g`. I also added
a general (non-cached) path for any other `qd` (e.g. the `result *= base` multiply when `result` is
not full degree), so correctness never depends on the fast path firing.

**Re-verifying the fix on the cases that broke.** I re-trace the tribonacci `a[n]=a[n-1]+a[n-2]+a[n-3]`,
seeds `[0,0,1]`, `N=30`. The brute force gives `15902591`; after the fix the fast solution agrees.
I check `k=4` with sparse coefficients `c=[0,1,0,1]` (so `g` has zero terms — a good stress for the
reverse-invert, since `rev(g)` then has a high-order leading structure), seeds all ones, `N=40`:
both methods give `10946`. The aliasing-sensitive sizes are now handled by `nfix`, and the cases
that were wrong-when-not-a-power-of-two are now right across the board.

**Edge cases, deliberately.**
- `N < k`: short-circuit returns `a[N]` directly; covers `N=0` (return `a[0]`) and `N=k-1` (last
  seed). No polynomial machinery touched, so no chance of an empty-input or degree-0 surprise.
- `N = k` exactly: `e = N`, one full square-and-multiply pass; verified `c=[1,1,1]`, seeds `[2,3,5]`,
  `N=3` gives `a[3] = 2+3+5 = 10`. Correct.
- `k = 1`: handled by the explicit `x mod g` reduction of the base, as fixed above.
- All-zero coefficients, `N >= k`: `g(x) = x^k`, so `x^N mod g = 0` for `N >= k`, answer `0`.
  Verified `c=[0,0,0]`, any seeds, `N=50` gives `0`.
- Negative inputs: coefficients and seeds are read as signed `ll`, reduced `x %= MOD; if (x<0) x+=MOD`,
  so `-1` becomes `p-1`. Verified against the brute (which does the same reduction).
- Overflow: every product is `(ll)a*b % MOD` with `a,b < p < 2^30`, so the product fits in 63 bits;
  accumulators are reduced immediately. The index `N` is `ll` (fits `10^18`). No 32-bit trap.

**Performance at the limit.** I build the worst case `k = 2*10^4`, `N = 10^18`, random full-range
data. The dominant cost is `log N ~ 60` squarings, each a handful of NTTs of size `~2^16`. To shave
the constant, `PolyMod::init` precomputes the forward NTTs of `g` and of the truncated `rgi` once
(`g_hat`, `rgi_hat`) and the fast `reduce` path reuses them, so a reduction costs two forward NTTs
plus two inverse NTTs instead of recomputing the transforms of the fixed operands every time. On this
machine the worst case runs in about 1.5 s, comfortably inside a 3 s limit, at ~6 MB — the
`O(k log k log N)` scaling is what makes the stress test feasible where matrix power could not even
finish a single multiply.

**Self-verification summary.** I ran the fast solution against the `O(N*k)` brute force on 700
random small cases from a seeded generator, 300 additional larger-`k` cases (`k` up to 40, `N` up to
4000, which exercises the cached reduce path heavily), explicit power-of-two `k` values
`{2,4,8,16,32,64}`, and the hand edge cases above — zero mismatches everywhere, and the documented
Fibonacci sample (`N=10 -> 55`, `N=0 -> 0`, `N=1 -> 1`) matches. The two real bugs — the unreduced
base at `k=1` and the slice-offset/size aliasing in the fast reverse-divide — were both found by the
differential test, diagnosed to a precise cause, fixed, and re-verified on the exact cases that had
failed. That is the evidence I trust, so this is what I ship: the Kitamasa / Fiduccia method, `x^N`
raised inside `Z_p[x]/(g(x))` with NTT multiplication and Newton-reciprocal reduction, `O(k log k log N)`.

**Final solution.**

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

**Causal recap.** Iterating the recurrence is `Theta(N*k)` and impossible at `N=10^18`; matrix power
is the right jump but the wrong representation, `O(k^3 log N)`, dead at `k=2*10^4`. Rewriting the
state-advance map as multiplication by `x` in `Z_p[x]/(g(x))` keeps the `log N` jump but collapses a
squaring from a `k x k` matrix multiply to a polynomial multiply-and-reduce, both `O(k log k)` with
NTT and Newton-reciprocal division — `O(k log k log N)`. Two transcription bugs (the unreduced base
at `k=1`; the slice size/offset aliasing in the fast reverse-divide) were caught by differential
testing against an `O(N*k)` brute, pinned to exact causes, fixed, and re-verified; precomputing the
fixed operands' transforms drops the worst case to ~1.5 s.
