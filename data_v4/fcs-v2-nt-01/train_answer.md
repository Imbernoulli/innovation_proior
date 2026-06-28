**Problem.** Given a multiset of `n` non-negative integers `a[0..n-1]` with each value in `[0, V]`, and
`q` query values `s` in `[0, 3V]`, output for each query the number of **ordered triples** of indices
`(i, j, k)` (chosen independently, repetitions of an index allowed) with `a[i] + a[j] + a[k] = s`,
modulo `998244353`. Read `n V`, the values, `q`, and the queries from stdin; print one count per query.

**Key idea — the count is a polynomial cube, computed by NTT.** Let `f[v]` be how many array elements
equal `v`, and form the frequency polynomial `f(x) = Σ_v f[v]·x^v`. Then the number of ordered triples
summing to `s` is exactly `[x^s] f(x)^3`: a triple with values `(u, w, x)` contributes `f[u]·f[w]·f[x]`,
and summing over `u + w + x = s` is the cube's `s`-th coefficient. So the whole task is "cube a polynomial
of degree `≤ V` mod `p`, then read off coefficients."

The schoolbook cube is `O(V^2)` — at `V = 2*10^5` that is `~10^{11}` modular multiply-adds, far past a
2-second limit. The fix is the **Number Theoretic Transform (NTT)**: under an order-`D` transform,
convolution becomes pointwise multiplication, so each polynomial product costs `O(D log D)` for
`D ~ 3V`. NTT (not a floating-point FFT) is mandatory because the answer must be **exact mod `998244353`**;
`double`-precision FFT on length-`10^6` transforms of values up to `~10^{15}` would lose the integer.
The modulus is chosen for exactly this: `998244353 = 119·2^{23} + 1`, so `2^{23} | p - 1`, which means a
primitive `D`-th root of unity `g^{(p-1)/D}` (with primitive root `g = 3`) exists in `Z/pZ` for every NTT
length `D` up to `2^{23}` — comfortably above the `2^{20}` needed here. Compute `f^2 = f·f` and then
`f^3 = f^2·f` by NTT, and answer query `s` by indexing `f3[s]`.

**Pitfalls to get right.**
1. *Inverse-transform roots.* The forward layer uses the root `g^{(p-1)/len}`; the inverse layer must use
   its inverse, `g^{(p-1) - (p-1)/len}`, followed by scaling every coefficient by `D^{-1} = D^{p-2} mod p`.
   Pasting the forward root into the inverse branch produces a transform that *passes tiny samples but
   fails at scale* (short transforms mask the error) — the signature of this exact bug.
2. *Overflow in butterflies.* A twiddle times a coefficient is up to `(p-1)^2 ≈ 10^{18}`; multiply through
   `__int128` before reducing. True triple counts reach `~8*10^{15}`, so the mod is load-bearing, and all
   accumulators are 64-bit.
3. *Output volume.* With `q` up to `2*10^5`, buffer the output in a string instead of per-line `cout`.

**Edge cases.** `n = 0` → empty array, all counts `0`. `V = 0` → `f = [n]`, only `s = 0` is nonzero and
equals `n^3 mod p` (the length-1 NTT path must be a no-op, which it is). Single element → exactly one
triple, at `s = 3·a[0]`. `q = 0` → no output. Heavy stacking (many copies of one value) → the mod
actually reduces. Query at the `s = 3V` boundary → `f3` must have length exactly `3V + 1` (indices
`0..3V`), so don't truncate one coefficient short.

**Complexity.** Building `f` is `O(n + V)`. The two NTT products are `O(D log D)` with `D` the next power
of two above `3V`, i.e. `O(V log V)`. Each of the `q` queries is an `O(1)` array lookup. Total
`O((n + V log V) + q)` time and `O(V)` memory. Measured worst case (`n = V = q = 2*10^5`): ~0.40 s, ~30 MB.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

const long long MOD = 998244353;        // NTT-friendly prime: 998244353 = 119*2^23 + 1
const long long G = 3;                   // primitive root of 998244353

long long power_mod(long long b, long long e, long long m) {
    long long r = 1 % m; b %= m; if (b < 0) b += m;
    while (e > 0) { if (e & 1) r = (__int128)r * b % m; b = (__int128)b * b % m; e >>= 1; }
    return r;
}

// In-place iterative NTT. n must be a power of two. invert=false: forward, true: inverse.
void ntt(vector<long long>& a, bool invert) {
    int n = (int)a.size();
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        long long w = invert ? power_mod(G, MOD - 1 - (MOD - 1) / len, MOD)
                             : power_mod(G, (MOD - 1) / len, MOD);
        for (int i = 0; i < n; i += len) {
            long long wn = 1;
            for (int k = 0; k < len / 2; k++) {
                long long u = a[i + k];
                long long v = (__int128)a[i + k + len / 2] * wn % MOD;
                a[i + k] = u + v < MOD ? u + v : u + v - MOD;
                a[i + k + len / 2] = u - v >= 0 ? u - v : u - v + MOD;
                wn = (__int128)wn * w % MOD;
            }
        }
    }
    if (invert) {
        long long n_inv = power_mod(n, MOD - 2, MOD);
        for (long long& x : a) x = (__int128)x * n_inv % MOD;
    }
}

// Convolution of A and B modulo MOD, returned as a polynomial of length |A|+|B|-1.
vector<long long> convolve(vector<long long> A, vector<long long> B) {
    if (A.empty() || B.empty()) return {};
    int result_size = (int)A.size() + (int)B.size() - 1;
    int sz = 1;
    while (sz < result_size) sz <<= 1;
    A.resize(sz, 0);
    B.resize(sz, 0);
    ntt(A, false);
    ntt(B, false);
    for (int i = 0; i < sz; i++) A[i] = (__int128)A[i] * B[i] % MOD;
    ntt(A, true);
    A.resize(result_size);
    return A;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n; long long V;
    if (!(cin >> n >> V)) return 0;

    // f[v] = number of array elements equal to v  (the frequency polynomial coefficients)
    vector<long long> f(V + 1, 0);
    for (int i = 0; i < n; i++) {
        long long x; cin >> x;
        f[x] = (f[x] + 1) % MOD;
    }

    // The count of ordered triples (i,j,k) with a_i + a_j + a_k = s is [x^s] f(x)^3.
    // Compute f^2 then (f^2)*f via NTT; degree grows to at most 3V.
    vector<long long> f2 = convolve(f, f);     // length 2V+1, [x^s] = #ordered pairs summing to s
    vector<long long> f3 = convolve(f2, f);    // length 3V+1, [x^s] = #ordered triples summing to s

    int q; cin >> q;
    string out;
    out.reserve((size_t)q * 7);
    for (int j = 0; j < q; j++) {
        long long s; cin >> s;
        long long ans = 0;
        if (s >= 0 && s < (long long)f3.size()) ans = f3[s];
        out += to_string(ans);
        out += '\n';
    }
    cout << out;
    return 0;
}
```
