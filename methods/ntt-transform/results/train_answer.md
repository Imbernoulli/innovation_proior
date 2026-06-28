The operation I care about is polynomial multiplication, and it hurts in two places that turn out to be the same shape. In R-LWE lattice cryptography every primitive step — encrypt, sign, key-exchange — reduces to multiplying two polynomials of degree below $n$ in $R_q = \mathbb{Z}_q[x]/(x^n + 1)$, with $n$ a power of two (a few hundred to a thousand) and $q$ a modest prime. And when I multiply two enormous integers, I chop each into base-$u$ digits and the product digits are a convolution of the digit vectors. Both are convolutions; schoolbook computes them with $O(n^2)$ coefficient products, and for $n = 1024$ that quadratic is the entire cost of the protocol. I want $O(n\log n)$, and — because the data are integers mod $q$ and the answer must be a bit-exact integer — I want it in exact modular arithmetic with no round-off.

The classical route to $O(n\log n)$ convolution is the Fourier transform: by the convolution theorem, transform both sequences, multiply pointwise, transform back. Over $\mathbb{C}$ this uses $\zeta = e^{2\pi i/n}$, the DFT $A_j = \sum_i a_i\,\zeta^{ij}$ and its inverse $a_i = \tfrac1n\sum_j A_j\,\zeta^{-ij}$, with Cooley–Tukey doing each transform in $n\log n$ instead of $n^2$. The trouble is purely arithmetic. If I run an integer convolution through complex floating-point FFTs, the output is a vector of floats that I round to the nearest integer at the end, and that rounding is correct only if the accumulated round-off across all those $e^{2\pi i/n}$ multiplications stays below $\tfrac12$. The precision needed for that grows with $n$ and with the coefficient magnitude; for a cryptographic kernel that also wants to be constant-time and bit-exact, threading a floating-point error budget through every multiply is exactly the fragility I want to avoid. The other baselines don't close the gap either: schoolbook is exact but quadratic; Karatsuba and Toom–Cook are exact and integer but their exponent stays above $1$ and never reaches quasi-linear; Nussbaumer's recursion avoids roots of unity but pads the inputs and carries overhead a root-of-unity transform avoids when a suitable prime exists. The transform is the right idea; doing it over $\mathbb{C}$ is the wrong arithmetic.

So the question is what the DFT actually needs from $\zeta$ — and the answer is: nothing complex. Summing the forward then the inverse, $\tfrac1n\sum_j\big(\sum_k a_k\zeta^{kj}\big)\zeta^{-ij} = \tfrac1n\sum_k a_k\sum_j \zeta^{(k-i)j}$, and the whole thing collapses to $a_i$ exactly because the inner sum $\sum_{j=0}^{n-1}\zeta^{Kj}$ equals $n$ when $K\equiv 0$ and $0$ otherwise. The vanishing is one line of telescoping: for $K\not\equiv 0\ (\mathrm{mod}\ n)$, $(1-\zeta^K)\sum_j\zeta^{Kj} = 1-\zeta^{Kn} = 0$, and since $\zeta^K\neq 1$ the factor $1-\zeta^K$ is nonzero, so the sum is zero. That derivation uses only three things about $\zeta$: that $\zeta^n = 1$, that $\zeta^K\neq 1$ for $0<K<n$, and that I can add, multiply, and divide. Order exactly $n$, plus a multiplicative inverse. No limits, no magnitude, no continuity. The complex numbers were just one place where an element of exact order $n$ happens to live.

I propose the Number-Theoretic Transform: the DFT carried out over a finite field $\mathbb{Z}_p$ instead of over $\mathbb{C}$. The nonzero elements of $\mathbb{Z}_p$ form a cyclic group of order $p-1$, so an element $\omega$ of order exactly $n$ exists precisely when $n \mid p-1$; it is obtained from a generator (primitive root) $g$ of $\mathbb{Z}_p^*$ as $\omega = g^{(p-1)/n}$. Such an $\omega$ is a primitive $n$-th root of unity, and $\omega^n = 1$ with $\omega^K\neq 1$ for $0<K<n$ — both conditions holding exactly in integer arithmetic mod $p$, with division available because $\mathbb{Z}_p$ is a field. Defining the transform exactly as the DFT but over $\mathbb{Z}_p$,
$$\hat A_j = \sum_{i=0}^{n-1} a_i\,\omega^{ij}\pmod q,\qquad a_i = n^{-1}\sum_{j=0}^{n-1}\hat A_j\,\omega^{-ij}\pmod q,$$
the same telescoping proof goes through verbatim with $\zeta$ replaced by $\omega$, so the convolution theorem holds and $(c_i) = \mathrm{INTT}\big(\mathrm{NTT}(a)\circ\mathrm{NTT}(b)\big)$ is the cyclic convolution — all in exact modular arithmetic, no round-off ever. The inverse carries the factor $n^{-1}$ because $M M^* = nI$ for $M_{ji}=\omega^{ij}$, $M^*_{ij}=\omega^{-ij}$: the $(i,k)$ entry of $M^*M$ is $\sum_j\omega^{(k-i)j} = n\,\delta_{ik}$ by the same identity, so $M^{-1} = \tfrac1n M^*$, and $n^{-1}$ here means the modular inverse of $n$.

The speed comes for free, because the Cooley–Tukey factorization is purely algebraic. Splitting the index by parity, with $E_j$ and $O_j$ the length-$n/2$ transforms of the even- and odd-indexed subsequences (using that $\omega^2$ is a primitive $(n/2)$-th root), gives
$$A_j = E_j + \omega^j O_j,\qquad A_{j+n/2} = E_j - \omega^j O_j,$$
where the sign flip in the second line is exactly $\omega^{n/2} = -1$ — the unique order-2 element of the group is $-1$. This is the complex butterfly with $\omega^j$ as twiddle, and nothing in it touched the complex numbers, so it transplants to $\mathbb{Z}_p$ unchanged: $\log_2 n$ stages of $n/2$ butterflies, $O(n\log n)$ modular operations.

That settles multiplication modulo $x^n - 1$ (cyclic convolution: overflow terms past degree $n-1$ fold back additively). But the cryptographic ring is $x^n + 1$, where $x^n\equiv -1$, so I want the negacyclic convolution $c_k = \sum_{i+j=k} a_i b_j - \sum_{i+j=k+n} a_i b_j$ — the wrapped terms folding back with a minus sign. Zero-padding to length $2n$ and reducing by hand works but doubles the transform and adds an explicit fold; I'd be paying $2n\log 2n$ plus a reduction pass for data that has only $n$ real coefficients. The better move is to bake $x^n + 1$ into a length-$n$ transform. The roots of $x^n + 1$ are the primitive $2n$-th roots of unity — the odd $2n$-th roots, whose $n$-th power is $-1$ — so I introduce $\psi$, a primitive $2n$-th root of unity in $\mathbb{Z}_q$ with $\psi^2 = \omega$ and $\psi^n = -1$; this exists exactly when $q\equiv 1\ (\mathrm{mod}\ 2n)$, a stronger congruence than the cyclic $q\equiv 1\ (\mathrm{mod}\ n)$, and is taken as $\psi = g^{(q-1)/2n}$. Now pre-weight the inputs by $\psi^i$. The cyclic convolution of $(\psi^i a_i)$ and $(\psi^i b_i)$ at index $k$ is
$$\sum_{i+j\equiv k} \psi^{i+j} a_i b_j = \psi^k\Big(\sum_{i+j=k} a_i b_j - \sum_{i+j=k+n} a_i b_j\Big) = \psi^k c_k,$$
because the un-wrapped terms carry $\psi^k$ while the wrapped terms carry $\psi^{k+n} = \psi^k\psi^n = -\psi^k$. The $\psi^n = -1$ is precisely what turns the additive wrap into a subtractive one — the sign of $x^n+1$ materializes out of the weighting. So I post-weight the result by $\psi^{-i}$, keyed on the *output* index (the weight lives outside the inverse sum: $\psi^{-i}$ keyed on $i$ roundtrips, $\psi^{-j}$ keyed on the summation index does not), and the negacyclic product falls out of a single length-$n$ transform — no padding to $2n$, no explicit reduction. Equivalently, the weighted forward transform is $\hat A_j = \sum_i \psi^{2ij+i} a_i = \sum_i \psi^{(2j+1)i} a_i$: it evaluates the polynomial at the odd powers $\psi^{2j+1}$, the primitive $2n$-th roots, the roots of $x^n+1$ — and evaluating at roots of $x^n+1$ *is* working modulo $x^n+1$, which is why the reduction is implicit.

The last refinement makes both the $\psi$-weighting and the bit-reversal cost nothing. Redoing the Cooley–Tukey split on the odd-power form $\hat A_j = \sum_i\psi^{(2j+1)i}a_i$, the even/odd inner sums become length-$n/2$ odd-power transforms with $\omega = \psi^2$ playing one level up, and the butterfly becomes $\hat A_j = A_j + \psi^{2j+1}B_j$, $\hat A_{j+n/2} = A_j - \psi^{2j+1}B_j$ (using $\psi^{2(j+n/2)+1} = -\psi^{2j+1}$). So the $\psi$-weighting is absorbed: I never pre-scale the inputs, I just use a twiddle table of odd $\psi$-powers rather than $\omega$-powers. A decimation-in-time forward transform takes standard order to bit-reversed order; the dual decimation-in-frequency inverse — the Gentleman–Sande butterfly $a[j]\leftarrow U+V$, $a[j+t]\leftarrow (U-V)\cdot S$ with $S$ a power of $\psi^{-1}$, where the $\psi^{-ni} = (-1)^i$ split keyed on output parity reproduces the negacyclic sign — takes bit-reversed order back to standard, and folds the $n^{-1}$ scaling in at the end. Pairing the two, the two bit-reversal permutations meet in the middle and cancel, since pointwise multiplication doesn't care about ordering, so no explicit bit-reversal pass is needed at all.

Concretely, the roots come from a primitive root $g$ of $\mathbb{Z}_q^*$ via $\psi = g^{(q-1)/2n}$ and $\omega = \psi^2 = g^{(q-1)/n}$, with $g$ found by the standard test (factor $q-1$; $g$ is a generator iff $g^{(q-1)/r}\neq 1$ for each prime $r\mid q-1$). The modulus is chosen of Proth form $q = k\cdot 2^m + 1$ with small $k$ and $2n\mid 2^m$, so that $q\equiv 1\ (\mathrm{mod}\ 2n)$ holds — in practice $q = 12289 = 3\cdot 2^{12}+1$ (BLISS / R-LWE key exchange) or $q = 7681$. That shape also gives a fast inner-loop reduction: since $k\cdot 2^m\equiv -1\ (\mathrm{mod}\ q)$, writing a product $C = C_0 + 2^m C_1$ gives $kC\equiv kC_0 - C_1$, a shift-multiply-subtract that changes the residue class by a tracked factor of $k$ to be divided out at the end (folded into the $n^{-1}$ scaling), with Montgomery and Barrett reduction as the general-purpose fallbacks. Checked on the smallest honest case — $n=4$, $q=7681$, $g=17$, $\omega=3383$, $\psi=1925$ — multiplying $[1,2,3,4]$ by $[5,6,7,8]$ modulo $x^4+1$ gives $[7625,7645,2,60]$ ($=[-56,-36,2,60]$ mod $7681$), exactly the schoolbook negacyclic product, and at the real parameters $n=512$, $q=12289$ random polynomials agree with schoolbook coefficient-for-coefficient.

The deliverable is a single-file C++17 program reading the instance from stdin: `q`, `n`, then the `n` coefficients of `f` and the `n` coefficients of `g`, printing the `n` coefficients of `f*g mod (x^n + 1)` reduced into `[0, q)`. It is the merged-`ψ` Cooley–Tukey forward / Gentleman–Sande inverse pair; every modular product runs through `__int128` so multiplying two near-`q` residues never overflows. (`q` a prime with `q ≡ 1 (mod 2n)`, `n` a power of two.)

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
