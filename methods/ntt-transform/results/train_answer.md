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

```python
import math

def find_primitive_root(q):
    """Generator g of (Z/qZ)^*, q prime."""
    phi = q - 1
    factors, m, d = set(), phi, 2
    while d * d <= m:
        if m % d == 0:
            factors.add(d)
            while m % d == 0:
                m //= d
        d += 1
    if m > 1:
        factors.add(m)
    for g in range(2, q):
        if all(pow(g, phi // f, q) != 1 for f in factors):
            return g
    raise ValueError("no primitive root")

def roots_of_unity(q, n):
    """psi = g^{(q-1)/2n} is a primitive 2n-th root (psi^n = -1); omega = psi^2."""
    g = find_primitive_root(q)
    psi = pow(g, (q - 1) // (2 * n), q)
    omega = pow(psi, 2, q)                         # = g^{(q-1)/n}, primitive n-th root
    assert pow(omega, n, q) == 1 and pow(psi, n, q) == q - 1
    return omega, psi

def brv(x, bits):
    r = 0
    for _ in range(bits):
        r = (r << 1) | (x & 1)
        x >>= 1
    return r

def _psi_table(psi, n, q):
    """psi^0..psi^{n-1} stored in bit-reversed index order (the twiddle table)."""
    bits = int(math.log2(n))
    pw = [pow(psi, i, q) for i in range(n)]
    return [pw[brv(i, bits)] for i in range(n)]

def ntt_forward(a, psi, q):
    """Negacyclic forward NTT, Cooley-Tukey butterflies (decimation-in-time).
    Input in standard order, output in bit-reversed order; psi weighting is built
    into the twiddle table, so there is no separate pre-scaling pass."""
    a = a[:]; n = len(a); Psi = _psi_table(psi, n, q)
    t, m = n, 1
    while m < n:
        t //= 2
        for i in range(m):
            j1 = 2 * i * t; S = Psi[m + i]
            for j in range(j1, j1 + t):
                U = a[j]; V = a[j + t] * S % q
                a[j] = (U + V) % q
                a[j + t] = (U - V) % q
        m *= 2
    return a

def ntt_inverse(a, psi, q):
    """Negacyclic inverse NTT, Gentleman-Sande butterflies (decimation-in-frequency).
    Input in bit-reversed order, output in standard order; the n^{-1} scaling and the
    psi^{-i} post-weighting are folded into the twiddle table and the final scale."""
    a = a[:]; n = len(a); Psi = _psi_table(pow(psi, -1, q), n, q)
    t, m = 1, n
    while m > 1:
        j1 = 0; h = m // 2
        for i in range(h):
            S = Psi[h + i]
            for j in range(j1, j1 + t):
                U = a[j]; V = a[j + t]
                a[j] = (U + V) % q
                a[j + t] = (U - V) * S % q
            j1 += 2 * t
        t *= 2; m //= 2
    ninv = pow(n, -1, q)
    return [x * ninv % q for x in a]

def negacyclic_mul(f, g, q):
    """Product f * g in Z_q[x]/(x^n + 1) via the NTT (n a power of two, q ≡ 1 mod 2n)."""
    n = len(f); _, psi = roots_of_unity(q, n)
    F = ntt_forward(f, psi, q); G = ntt_forward(g, psi, q)
    H = [F[i] * G[i] % q for i in range(n)]
    return ntt_inverse(H, psi, q)

# --- definitional O(n^2) transforms, for the cyclic case and cross-checking ---
def ntt_naive(a, omega, q):
    n = len(a)
    return [sum(a[i] * pow(omega, i * j, q) for i in range(n)) % q for j in range(n)]

def intt_naive(A, omega, q):
    n = len(A); ninv, oinv = pow(n, -1, q), pow(omega, -1, q)
    return [ninv * sum(A[j] * pow(oinv, i * j, q) for j in range(n)) % q for i in range(n)]

def schoolbook_negacyclic(f, g, q):
    n = len(f); res = [0] * n
    for i in range(n):
        for j in range(n):
            k = i + j
            if k < n: res[k] += f[i] * g[j]
            else:     res[k - n] -= f[i] * g[j]     # x^n = -1
    return [x % q for x in res]

if __name__ == "__main__":
    # toy: n=4, q=7681 (= 15*2^9 + 1).  g=17, omega=3383, psi=1925.
    q, n = 7681, 4
    omega, psi = roots_of_unity(q, n)
    f, g = [1, 2, 3, 4], [5, 6, 7, 8]
    assert intt_naive(ntt_naive(f, omega, q), omega, q) == f       # cyclic identity
    assert negacyclic_mul(f, g, q) == schoolbook_negacyclic(f, g, q) == [7625, 7645, 2, 60]
    # real parameters: n=512, q=12289 (= 3*2^12 + 1), random polynomials
    import random
    q, n = 12289, 512; random.seed(0)
    f = [random.randrange(q) for _ in range(n)]
    g = [random.randrange(q) for _ in range(n)]
    assert negacyclic_mul(f, g, q) == schoolbook_negacyclic(f, g, q)
    print("NTT negacyclic convolution matches schoolbook")
```
