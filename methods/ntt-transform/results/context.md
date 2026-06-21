# Context

## Research question

Polynomial multiplication is the bottleneck operation in two large families of computation. The first is exact arithmetic on long objects: multiplying two big integers, or two polynomials of high degree with integer coefficients, where the schoolbook method costs `O(n^2)` coefficient multiplications. The second is ideal lattice-based cryptography built on the Ring Learning With Errors (R-LWE) problem, where every encryption, signature, or key-exchange step multiplies two polynomials in a quotient ring `R_q = Z_q[x]/(x^n + 1)` with `n` a power of two (typically `256`, `512`, or `1024`) and `q` a modest prime. Here the product is taken modulo `x^n + 1`, so the natural operation is not an ordinary product but a **negacyclic convolution** of the two coefficient vectors.

The goal is to compute these products in quasi-linear time `O(n log n)` instead of `O(n^2)` using exact integer (modular) arithmetic, with no floating-point round-off. The precise problem: given `n` a power of two and a prime `q` with the right structure, multiply two degree-`< n` polynomials over `Z_q` modulo `x^n + 1` in `O(n log n)` modular operations, exactly.

## Background

**The Discrete Fourier Transform and the convolution theorem.** For a length-`n` sequence `(a_i)` over the complex numbers, the DFT is `A_j = Σ_i a_i ζ^{ij}` with `ζ = e^{2πi/n}` an `n`-th root of unity. Its inverse is `a_i = (1/n) Σ_j A_j ζ^{-ij}`. The single property that makes it useful for multiplication is the **convolution theorem**: the DFT turns convolution into pointwise product. If `(a_i)`, `(b_i)` are length-`n` sequences and `(c_i)` is their cyclic convolution `c_k = Σ_{i+j≡k (mod n)} a_i b_j`, then `C_j = A_j B_j`. So a cyclic convolution can be computed by two forward transforms, `n` scalar products, and one inverse transform. Cyclic convolution of coefficient vectors is exactly polynomial multiplication modulo `x^n - 1`.

**The fast Fourier transform.** Computing the DFT by the matrix definition costs `O(n^2)`. The Cooley–Tukey algorithm (Cooley & Tukey 1965) reduces it to `O(n log n)` by a divide-and-conquer factorization. In the radix-2 decimation-in-time form, split the input by index parity: with `E_k` the DFT of the even-indexed subsequence and `O_k` the DFT of the odd-indexed one, `A_k = E_k + ζ^k O_k` and `A_{k+n/2} = E_k - ζ^k O_k` for `0 ≤ k < n/2`. The `ζ^k` is the *twiddle factor*; the paired `±` combination is the *butterfly*. The sign flip in the second line uses `ζ^{n/2} = -1`. Recursing gives `log_2 n` stages of `n/2` butterflies each. The general 1965 formulation handles any composite `n = n_1 n_2` by reindexing the data as a 2-D array and transforming along each axis with twiddle factors in between; the radix-2 case is the special case `n_1 = 2`.

**Roots of unity in a finite field.** The DFT and the convolution theorem are proved from one purely algebraic fact about `ζ`: for any integer `K`, `Σ_{i=0}^{d-1} ζ^{iK} = d` if `K ≡ 0 (mod d)` and `0` otherwise — the second case because `(1 - ζ^K) Σ_i ζ^{iK} = 1 - ζ^{Kd} = 0` while `1 - ζ^K ≠ 0`. Nothing here uses that `ζ` is complex; it needs only an element of multiplicative order `d` in some field, and that `1 - ζ^K ≠ 0` for `K ≢ 0 (mod d)`. A finite field `GF(p^n)` has a cyclic multiplicative group of order `p^n - 1`, so it contains an element `r` of order `d` for every `d | p^n - 1`. For the prime field `Z_p` this means: an element of order `n` exists exactly when `n | p - 1`. Such an `r` is a **primitive `n`-th root of unity** in `Z_p`, and it can be produced from a primitive root (generator) `w` of `Z_p^*` as `r = w^{(p-1)/n}`. A generator `w` is detected by the test `w^{(p^n-1)/q} ≠ 1` for every prime `q | p^n - 1`.

**Negacyclic versus cyclic.** A length-`n` transform built from an `n`-th root of unity computes convolution modulo `x^n - 1` (cyclic). The cryptographic ring uses `x^n + 1`, the `2n`-th cyclotomic polynomial for `n` a power of two. Convolution modulo `x^n + 1` is the **negacyclic** (negative-wrapped) convolution: `c_k = Σ_{i+j=k} a_i b_j - Σ_{i+j=k+n} a_i b_j`, where the terms that "wrap past degree `n`" enter with a minus sign instead of folding back additively. The wrap sign is the only difference from the cyclic case, and it mirrors the relation `x^n ≡ -1 (mod x^n + 1)`.

**Modulus structure and modular reduction.** An `n`-th root of unity exists in `Z_q` exactly when `q ≡ 1 (mod n)`; a `2n`-th root exactly when `q ≡ 1 (mod 2n)`, the stronger congruence. Historically the convenient choices were Fermat primes `2^{2^k}+1`, for which `q - 1 = 2^{2^k}` is a pure power of two so plenty of `2`-power roots of unity live in `Z_q`, and Mersenne primes `2^p - 1`, where `2^p ≡ 1 (mod q)` makes reduction modulo `q` cheap (note `q - 1 = 2^p - 2 = 2(2^{p-1}-1)` carries only a single factor of two, so Mersenne primes are chosen for cheap reduction rather than for an abundance of radix-2 roots). More generally, primes of Proth form `q = k·2^m + 1` with small `k` make `q ≡ 1 (mod 2n)` easy to satisfy whenever `2n | 2^m`. The inner loop of any modular transform is dominated by reduction modulo `q` after each multiply; the standard general-purpose techniques are **Montgomery reduction** (work in a residue system `x̃ = x·R mod q` so reduction of a product is a multiply-add-shift with no division) and **Barrett reduction** (replace the division in `⌊x/q⌋` by a precomputed reciprocal). For a modulus of the special form `q = k·2^m + 1`, the relation `k·2^m ≡ -1 (mod q)` lets one split `C = C_0 + 2^m C_1` and reduce via `kC ≡ kC_0 - C_1 (mod q)`, a shift-and-subtract that changes the residue class by a tracked factor of `k`, in the spirit of Montgomery's representation.

**Multi-prime CRT approach.** For the cyclic case, a single prime `p` with `p ≡ 1 (mod d)` and `p` larger than twice the largest convolution coefficient suffices; otherwise several primes `p_i ≡ 1 (mod d)` — all supporting the same transform length `d` — with product `P = Π p_i` exceeding the coefficient bound are used, and the result is reassembled coefficient-wise by the Chinese Remainder Theorem (Pollard 1971, §3).

## Baselines

**Schoolbook polynomial multiplication.** Multiply term by term, `O(n^2)` coefficient multiplications, then reduce the degree-`(2n-2)` result modulo `x^n ± 1` by a long division (for `x^n + 1`, fold the high half back with a sign flip). Exact and simple, and quadratic in `n`.

**Complex floating-point FFT convolution.** Apply a Cooley–Tukey FFT over `C` to both inputs, multiply pointwise, inverse-FFT, and round to the nearest integer. Achieves `O(n log n)`, and is the workhorse for signal processing and was suggested for the ring-based scheme NTRUEncrypt.

**Karatsuba / Toom–Cook splitting.** Recursively split each operand into parts and trade multiplications for additions, giving sub-quadratic exponents (`n^{log_2 3}` for Karatsuba). Exact and integer, useful at small-to-moderate `n`.

**Nussbaumer's algorithm.** Computes negacyclic convolutions recursively without needing roots of unity in the coefficient ring, by working over an auxiliary polynomial ring.

## Evaluation settings

The natural yardsticks are the two application regimes. For big-integer / big-polynomial multiplication: pairs of operands of increasing size (e.g. integers of thousands to millions of bits, encoded base `u` into coefficient vectors), measuring correctness against schoolbook or library multiplication and runtime as a function of length; when a single prime is too small, several primes `p_i ≡ 1 (mod d)` for a common transform length `d` are used with CRT recombination. For R-LWE cryptography: the standardized parameter sets — `n = 256, 512, 1024` with primes such as `q = 12289 = 3·2^{12}+1` (BLISS signatures, R-LWE key exchange) and `q = 7681` or `q = 8380417` — multiplying uniformly random and error-distributed polynomials in `Z_q[x]/(x^n+1)`. Metrics: bit-exact agreement of the negacyclic product with schoolbook, transform throughput (cycles, or modular multiplies per transform), and constant-time behaviour. A faithful re-implementation needs only exact integer arithmetic modulo `q`; no floating point is required, which is the point.

## Code framework

The pieces already on hand: exact modular arithmetic in `Z_q` (`+`, `-`, `*`, and modular inverse via extended Euclid or Fermat), a routine to find a primitive root of `Z_q^*` and hence roots of unity, a bit-reversal permutation, and a schoolbook convolution to check against. The radix-2 Cooley–Tukey butterfly is known as a complex-arithmetic primitive.

```python
import math

def find_primitive_root(q):
    """Generator of (Z/qZ)^* for prime q — already a standard routine."""
    phi = q - 1
    factors = set()
    m, d = phi, 2
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

def brv(x, bits):
    """Bit-reversal of x in 'bits' bits — a standard permutation primitive."""
    r = 0
    for _ in range(bits):
        r = (r << 1) | (x & 1)
        x >>= 1
    return r

def ring_multiply(f, g, q):
    # TODO: multiply f*g in Z_q[x]/(x^n + 1) exactly, in O(n log n) modular operations.
    pass

def schoolbook_negacyclic(f, g, q):
    n = len(f)
    res = [0] * n
    for i in range(n):
        for j in range(n):
            k = i + j
            if k < n:
                res[k] += f[i] * g[j]
            else:
                res[k - n] -= f[i] * g[j]   # x^n = -1
    return [x % q for x in res]
```
