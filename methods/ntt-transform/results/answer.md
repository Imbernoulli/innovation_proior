# Number-Theoretic Transform and fast negacyclic convolution

## Problem

Multiply two degree-`< n` polynomials in `O(n log n)` modular operations, exactly (no floating-point round-off), where the product is taken in `Z_q[x]/(x^n + 1)` with `n` a power of two and `q` a prime — the workhorse operation of R-LWE lattice cryptography, and (via a zero-padded cyclic NTT, with CRT over several primes when one is too small) of big-integer multiplication. The product modulo `x^n + 1` is the **negacyclic convolution** of the coefficient vectors: `c_k = Σ_{i+j=k} a_i b_j − Σ_{i+j=k+n} a_i b_j`.

## Key idea

The DFT and the convolution theorem need only one algebraic fact about the root `ζ`: it has exact order `n` and `1 − ζ^K ≠ 0` for `0 < K < n`, so that `Σ_{j} ζ^{Kj} = n·[K≡0 (mod n)]`. The complex `e^{2πi/n}` is not essential. A finite field `Z_p` has a cyclic group `Z_p^*` of order `p−1`, hence a **primitive `n`-th root of unity** `ω` exactly when `n | p−1` — obtained as `ω = g^{(p−1)/n}` for a generator `g`. Running the DFT over `Z_p` with `ω` is the **NTT**:

`Â_j = Σ_i a_i ω^{ij} (mod q)`,  `a_i = n^{−1} Σ_j Â_j ω^{−ij} (mod q)`,

both exact, no round-off. The inverse carries `n^{−1}` because `M M^* = n I` (same telescoping identity). The Cooley–Tukey radix-2 butterfly is purely algebraic — it uses `ω^{n/2} = −1` — so it transplants to `Z_q` verbatim, giving `O(n log n)`:

`A_j = E_j + ω^j O_j`,  `A_{j+n/2} = E_j − ω^j O_j`,

with `E, O` the NTTs of the even/odd-indexed subsequences. Then cyclic convolution (product mod `x^n − 1`) is `INTT(NTT(a) ∘ NTT(b))`.

**Negacyclic via the ψ trick.** The cryptographic ring is `x^n + 1`, not `x^n − 1`. Let `ψ` be a primitive `2n`-th root of unity with `ψ^2 = ω` and `ψ^n = −1` (exists when `q ≡ 1 (mod 2n)`; take `ψ = g^{(q−1)/2n}`). Pre-weight the inputs by `ψ^i`. The cyclic convolution of `(ψ^i a_i)` and `(ψ^i b_i)` at index `k` is `Σ_{i+j≡k} ψ^{i+j} a_i b_j = ψ^k (Σ_{i+j=k} a_i b_j − Σ_{i+j=k+n} a_i b_j) = ψ^k c_k`, since the wrap term picks up `ψ^{k+n} = −ψ^k`. So post-weight by `ψ^{−i}` (keyed on the output index) and the **negacyclic** product falls out of a single length-`n` transform — no zero-padding to `2n`, no explicit reduction. Equivalently the weighted forward transform is `Â_j = Σ_i ψ^{(2j+1)i} a_i`: it evaluates at the odd `2n`-th roots, which are the roots of `x^n + 1`, so reducing mod `x^n + 1` is implicit.

**Merged ψ, no bit-reversal.** Folding the `ψ^i`/`ψ^{−i}` weights into the twiddle factors (so the weighted transform evaluates at the odd `2n`-th roots `ψ^{2j+1}` — the roots of `x^n + 1` — with stage twiddles drawn from a bit-reversed table of `ψ`-powers rather than `ω`-powers) makes the pre/post weighting free. A decimation-in-time **Cooley–Tukey** forward transform takes standard order to bit-reversed order; a decimation-in-frequency **Gentleman–Sande** inverse takes bit-reversed to standard. Pairing them cancels the bit-reversal permutation, since pointwise multiplication ignores ordering.

## Algorithm

```
Given prime q ≡ 1 (mod 2n), n a power of two:
  g  = generator of Z_q^*;  psi = g^{(q-1)/2n};  omega = psi^2
  Precompute Psi   = powers of psi   in bit-reversed order
             PsiInv= powers of psi^-1 in bit-reversed order
  NTT (Cooley-Tukey, standard -> bit-reversed):
     for each stage, butterfly  U=a[j], V=a[j+t]*S; a[j]=U+V, a[j+t]=U-V   (S in Psi)
  INTT (Gentleman-Sande, bit-reversed -> standard):
     for each stage, butterfly  U=a[j], V=a[j+t]; a[j]=U+V, a[j+t]=(U-V)*S (S in PsiInv)
     then scale every coefficient by n^{-1}
  Negacyclic product f*g mod (x^n+1):
     F=NTT(f); G=NTT(g);  H=F∘G (pointwise);  return INTT(H)
Modulus: Proth prime q = k·2^m + 1 (e.g. 12289 = 3·2^12+1). Reduction: Montgomery /
Barrett in general; or K-RED using k·2^m ≡ -1 (mod q): K-RED(C)=k·C0 - C1 with C=C0+2^m C1.
```

## Code

Single-file C++17, reading the instance from stdin. It reads `q`, `n`, then the `n` coefficients of `f` and the `n` coefficients of `g`, and prints the `n` coefficients of `f*g mod (x^n + 1)` reduced into `[0, q)` — the merged-`ψ` Cooley–Tukey forward / Gentleman–Sande inverse pair, with all modular products carried through `__int128` so a multiply of two near-`q` residues never overflows. (`q` a prime with `q ≡ 1 (mod 2n)`, `n` a power of two.)

```cpp
// Negacyclic polynomial multiplication in Z_q[x]/(x^n + 1) via the NTT, O(n log n).
// stdin:  q  n   then n coefficients of f   then n coefficients of g
//         (n a power of two, q a prime with q == 1 (mod 2n))
// stdout: the n coefficients of f*g mod (x^n + 1), each reduced into [0, q).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

ll power_mod(ll a, ll e, ll q) {            // a^e mod q, exact via 128-bit products
    a %= q; if (a < 0) a += q;
    ll r = 1 % q;
    while (e > 0) {
        if (e & 1) r = (__int128)r * a % q;
        a = (__int128)a * a % q;
        e >>= 1;
    }
    return r;
}

ll inv_mod(ll a, ll q) { return power_mod(a, q - 2, q); }   // q prime => Fermat inverse

// Generator g of (Z/qZ)^*, q prime: factor q-1 and test orders.
ll find_primitive_root(ll q) {
    ll phi = q - 1, m = phi;
    vector<ll> factors;
    for (ll d = 2; d * d <= m; ++d)
        if (m % d == 0) { factors.push_back(d); while (m % d == 0) m /= d; }
    if (m > 1) factors.push_back(m);
    for (ll g = 2; g < q; ++g) {
        bool ok = true;
        for (ll f : factors) if (power_mod(g, phi / f, q) == 1) { ok = false; break; }
        if (ok) return g;
    }
    throw runtime_error("no primitive root");
}

int brv(int x, int bits) {                  // bit-reversal of x in 'bits' bits
    int r = 0;
    for (int i = 0; i < bits; ++i) { r = (r << 1) | (x & 1); x >>= 1; }
    return r;
}

// psi^0..psi^{n-1} stored in bit-reversed index order (the twiddle table).
vector<ll> psi_table(ll psi, int n, ll q) {
    int bits = __builtin_ctz((unsigned)n);
    vector<ll> pw(n);
    pw[0] = 1 % q;
    for (int i = 1; i < n; ++i) pw[i] = (__int128)pw[i - 1] * psi % q;
    vector<ll> rev(n);
    for (int i = 0; i < n; ++i) rev[i] = pw[brv(i, bits)];
    return rev;
}

// Negacyclic forward NTT, Cooley-Tukey (decimation-in-time): standard -> bit-reversed.
// psi weighting is built into the twiddle table, so there is no separate pre-scale.
void ntt_forward(vector<ll> &a, ll psi, ll q) {
    int n = a.size();
    vector<ll> Psi = psi_table(psi, n, q);
    for (int m = 1, t = n; m < n; m <<= 1) {
        t >>= 1;
        for (int i = 0; i < m; ++i) {
            int j1 = 2 * i * t; ll S = Psi[m + i];
            for (int j = j1; j < j1 + t; ++j) {
                ll U = a[j], V = (__int128)a[j + t] * S % q;
                a[j]     = (U + V) % q;
                a[j + t] = (U - V % q + q) % q;
            }
        }
    }
}

// Negacyclic inverse NTT, Gentleman-Sande (decimation-in-frequency): bit-reversed -> standard.
// The n^{-1} scaling and psi^{-i} post-weighting fold into the twiddle table and final scale.
void ntt_inverse(vector<ll> &a, ll psi, ll q) {
    int n = a.size();
    vector<ll> Psi = psi_table(inv_mod(psi, q), n, q);
    for (int t = 1, m = n; m > 1; t <<= 1, m >>= 1) {
        int j1 = 0, h = m / 2;
        for (int i = 0; i < h; ++i) {
            ll S = Psi[h + i];
            for (int j = j1; j < j1 + t; ++j) {
                ll U = a[j], V = a[j + t];
                a[j]     = (U + V) % q;
                a[j + t] = (__int128)((U - V + q) % q) * S % q;
            }
            j1 += 2 * t;
        }
    }
    ll ninv = inv_mod(n % q, q);
    for (int i = 0; i < n; ++i) a[i] = (__int128)a[i] * ninv % q;
}

// Product f*g in Z_q[x]/(x^n + 1): forward-NTT both, multiply pointwise, inverse-NTT.
vector<ll> negacyclic_mul(vector<ll> f, vector<ll> g, ll q) {
    int n = f.size();
    ll root = find_primitive_root(q);
    ll psi = power_mod(root, (q - 1) / (2LL * n), q);   // primitive 2n-th root, psi^n = -1
    ntt_forward(f, psi, q);
    ntt_forward(g, psi, q);
    vector<ll> h(n);
    for (int i = 0; i < n; ++i) h[i] = (__int128)f[i] * g[i] % q;
    ntt_inverse(h, psi, q);
    return h;
}

int main() {
    ios_base::sync_with_stdio(false); cin.tie(nullptr);
    ll q; int n;
    if (!(cin >> q >> n)) return 0;
    vector<ll> f(n), g(n);
    for (int i = 0; i < n; ++i) { cin >> f[i]; f[i] = ((f[i] % q) + q) % q; }
    for (int i = 0; i < n; ++i) { cin >> g[i]; g[i] = ((g[i] % q) + q) % q; }
    vector<ll> c = negacyclic_mul(f, g, q);
    for (int i = 0; i < n; ++i) cout << c[i] << (i + 1 < n ? ' ' : '\n');
    return 0;
}
```

`[7625, 7645, 2, 60] (= [−56, −36, 2, 60] mod 7681)` is the schoolbook product of `1+2x+3x^2+4x^3` and `5+6x+7x^2+8x^3` modulo `x^4 + 1`; the fast NTT reproduces it exactly, and agrees with schoolbook on random length-512 inputs at the BLISS/key-exchange prime `q = 12289`.
