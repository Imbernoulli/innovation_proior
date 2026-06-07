# Schönhage–Strassen integer multiplication

## Problem

Multiply two n-bit integers exactly in far fewer than the Θ(n²) bit operations of the
schoolbook method — ideally near-linear — with no floating-point round-off, so the result is
correct to the last bit, and with a structure that composes with itself at every scale.

## Key idea

Multiplication is convolution: writing each operand in base 2^M as a digit sequence, the
product's digits (before carries) are the convolution of the two sequences. A convolution
diagonalizes under the Fourier transform — `a∗b = IDFT(DFT(a)·DFT(b))` — and the FFT computes
each transform in Θ(N log N) ring operations. To make the transform *exact*, run it not over ℂ
but over the Fermat ring **R = Z/(2^n+1)Z**. There 2^n ≡ −1, so 2 has order 2n and
**ω = 2^{2n/N}** is a primitive N-th root of unity that is *a power of two* — every twiddle, every
weight, and the 1/N of the inverse transform become bit-shifts (with a cheap mod-(2^n+1) fold,
since bits past position n wrap back with a sign flip). The transform is multiplication-free; the
only genuine multiplications are the N pointwise products of n-bit residues.

Weighting the inputs by **θ^i = 2^{(n/N)i}** (a 2N-th root, θ^N = 2^n ≡ −1) before a plain cyclic
NTT, and unweighting by θ^{−k} afterward, converts the cyclic convolution into the **negacyclic**
one: the result lands modulo (2^{MN}+1), a Fermat-type modulus of the same shape as the inner
ring. Hence each of the N pointwise products is a smaller instance of the identical
"multiply mod 2^something+1" problem — **recurse**. With piece size q0 = Θ(√q) the modular cost obeys

  F(q) ≤ (2q/q0)·F(q0) + O(q log q),  q0 = Θ(√q),

so each level adds only O(1) to F(q)/(q log q); the size falls q → √q → q^{1/4} → … and reaches the
base case after **Θ(log log q)** levels, giving

  **T(n) = O(n log n log log n)** bit operations, in exact integer arithmetic.

(The original construction realizes R as a genuine Fermat number F_n = 2^{2ⁿ}+1, reducing
multiplication in Z_{F_m} to 2n multiplications in Z_{F_n} with m = 2n−1 or 2n−2 and root
w_{n+1}=2; a complex-FFT variant over ℂ gives the slightly weaker O(n log n (log log n)^{1+ε}) and
serves as the stepping stone before the Fermat ring removes the round-off.)

## Parameters (correctness constraints)

- N = 2^k pieces of M = S/N bits each; treat the integer as a degree-(N−1) polynomial at x = 2^M.
- Ring width n ≥ 2M + log₂N + 1, so each coefficient c_k = Σ a_i b_j < N·2^{2M} fits in Z/(2^n+1)Z.
- N | n, so both θ = 2^{n/N} (negacyclic 2N-th root) and ω = θ² = 2^{2n/N} (N-th root) exist as
  powers of two; N⁻¹ = 2^{−k} ≡ 2^{k(2n−1)} mod (2^n+1) is also a shift.
- Negacyclic evaluates only at the odd powers θ, θ³, …, θ^{2N−1} (the roots of x^N+1), so exactly N
  pointwise products are needed — half of what a zero-padded length-2N cyclic transform would use.
- Top level: get the full product by filling only the lower N/2 pieces (wrap lands in zeros) and
  choosing MN ≥ 2S, so 2^{MN}+1 exceeds the true product.
- Optional √2 trick: for n divisible by 4, (2^{3n/4} − 2^{n/4})² ≡ 2, a root of order 4n, doubling
  the usable transform length for the same ring (multiplication by √2 = two shifts and a subtract).

## Algorithm

1. Split a, b into N = 2^k pieces of M bits.
2. Weight piece i by θ^i = 2^{(n/N)i} (a shift) — negacyclic.
3. Forward NTT both (radix-2 butterflies; each "×ω^j" is a shift-and-fold).
4. Pointwise-multiply the N coefficient pairs mod 2^n+1 — recurse here for large n.
5. Inverse NTT (twiddles ω^{−j}, then ×N⁻¹ — all shifts).
6. Unweight position k by θ^{−k} (a shift).
7. Reassemble Σ c_k 2^{Mk} with carry propagation, reduced mod 2^{MN}+1.

## Code (illustrative, exact; same pipeline a production fast-multiply uses)

```python
# Arithmetic in R = Z/(2^n + 1)Z.  In R: 2^n == -1, so 2 has order 2n,
# and ω = 2^(2n/N), θ = 2^(n/N), 1/N = 2^(-log2 N) are all powers of two => shifts.

def reduce_mod(x, n):
    return x % ((1 << n) + 1)

def shift_mod(x, exponent, n):
    # multiply by 2^exponent in R; exponent is taken modulo the order 2n.
    return reduce_mod(x << (exponent % (2 * n)), n)

def ntt(a, n, inverse=False):
    # length-N cyclic NTT, omega = 2^(2n/N); every '* w^j' is a shift.
    N = len(a)
    assert N & (N - 1) == 0 and (2 * n) % N == 0
    j = 0                                                # bit-reversal permutation
    for i in range(1, N):
        bit = N >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    length = 2
    while length <= N:
        step = (2 * n) // length                         # root for this stage
        if inverse:
            step = -step                                 # inverse root
        for start in range(0, N, length):
            twiddle = 0
            for i in range(length // 2):
                u = a[start + i]
                v = shift_mod(a[start + i + length // 2], twiddle, n)
                a[start + i]               = reduce_mod(u + v, n)   # butterfly
                a[start + i + length // 2] = reduce_mod(u - v, n)
                twiddle += step
        length <<= 1
    if inverse:                                           # 1/N = 2^{-log2 N}
        k = N.bit_length() - 1
        for i in range(N):
            a[i] = shift_mod(a[i], -k, n)
    return a

def multiply_mod(a, b, K, M, n):
    # a*b mod (2^(K*M)+1): K=2^k pieces of M bits, ring 2^n+1; needs K|n, n>=2M+log2 K+1.
    out_n = K * M
    out_mod = (1 << out_n) + 1
    a %= out_mod
    b %= out_mod
    if a == (1 << out_n) or b == (1 << out_n):
        return (a * b) % out_mod                          # the residue that does not fit in out_n bits
    pa = [(a >> (M * i)) & ((1 << M) - 1) for i in range(K)]
    pb = [(b >> (M * i)) & ((1 << M) - 1) for i in range(K)]
    theta_step = n // K                                   # theta = 2^theta_step, theta^K = 2^n == -1
    A = [shift_mod(pa[i], theta_step * i, n) for i in range(K)]   # negacyclic weights
    B = [shift_mod(pb[i], theta_step * i, n) for i in range(K)]
    ntt(A, n); ntt(B, n)
    C = [pointwise_mul(A[i], B[i], n) for i in range(K)]  # recursive pointwise products
    ntt(C, n, inverse=True)
    C = [shift_mod(C[i], -theta_step * i, n) for i in range(K)]   # unweight
    res = 0
    for i in range(K):
        ci = C[i]
        if ci > (1 << (n - 1)): ci -= (1 << n) + 1        # negacyclic coeffs may be negative
        res += ci << (M * i)                              # reassembly with carry propagation
    return res % out_mod

RECURSE_BITS = 1 << 14
def pointwise_mul(x, y, n):
    if n <= RECURSE_BITS:
        return reduce_mod(x * y, n)                       # base case: stop recursing
    if x == (1 << n) or y == (1 << n):
        return reduce_mod(x * y, n)                       # residue -1 does not fit in n split bits
    k = max(2, n.bit_length() // 2)
    K = 1 << k                                            # recurse: same routine, smaller n
    while K > 2 and n % K:
        k -= 1
        K = 1 << k
    if n % K:
        return reduce_mod(x * y, n)
    M = n // K
    np = 2 * M + K.bit_length()                           # 2M + log2(K) + 1
    np = ((np + K - 1) // K) * K                          # round so K | n'
    if np >= n:
        return reduce_mod(x * y, n)
    return multiply_mod(x, y, K, M, np) % ((1 << n) + 1)

def multiply(a, b):
    # full product via zero-padded negacyclic transform (data fills lower K/2 pieces).
    if a == 0 or b == 0:
        return 0
    sign = -1 if (a < 0) ^ (b < 0) else 1
    a, b = abs(a), abs(b)
    S = max(a.bit_length(), b.bit_length())
    k = max(2, (2 * S).bit_length().bit_length() + 2)
    K = 1 << k
    M = (S + (K // 2) - 1) // (K // 2)
    np = 2 * M + K.bit_length()
    np = ((np + K - 1) // K) * K
    return sign * multiply_mod(a, b, K, M, np)
```

The transform is multiplication-free (only shifts, adds, and the cheap 2^n+1 fold); the genuine
work is the N pointwise products, each a smaller instance of the same routine, so the cost is
O(n log n log log n) — and because the ring is finite and the root of unity is a power of two,
every step is exact integer arithmetic.
