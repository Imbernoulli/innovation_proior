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

`[7625, 7645, 2, 60] (= [−56, −36, 2, 60] mod 7681)` is the schoolbook product of `1+2x+3x^2+4x^3` and `5+6x+7x^2+8x^3` modulo `x^4 + 1`; the fast NTT reproduces it exactly, and agrees with schoolbook on random length-512 inputs at the BLISS/key-exchange prime `q = 12289`.
