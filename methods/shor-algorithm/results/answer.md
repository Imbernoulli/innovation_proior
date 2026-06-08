# Shor's algorithm — polynomial-time factoring via quantum order-finding

## Problem

Factor an `l`-bit integer `N` in time polynomial in `l = log N`. Every classical method is superpolynomial (number field sieve: `exp(c (log N)^{1/3}(log log N)^{2/3})`). RSA's security rests on this hardness.

## Key idea

Factoring reduces to **order-finding** — computing the period `r` of `a → x^a mod N` (the least `r` with `x^r ≡ 1 (mod N)`) — and a quantum computer finds that period efficiently by interference. Prepare a uniform superposition over the exponent `a`, compute `x^a mod N` (making the second register periodic in `a` with period `r`), apply the **quantum Fourier transform** to the exponent register, and measure. The QFT makes the amplitudes constructively interfere at `c ≈ d·q/r`; with noticeable probability a measurement yields a `c` with `|c/q − d/r| ≤ 1/(2q)`, and a classical **continued-fraction** expansion recovers `r` when `gcd(d, r) = 1`. With `r`, `gcd(x^{r/2} ± 1, N)` gives a factor.

## The classical reduction (Miller)

For odd `N` that is not a prime power: choose random `x`, `1 < x < N`.
- If `gcd(x, N) ≠ 1`, that gcd is a factor.
- Otherwise find the order `r`. If `r` is even and `x^{r/2} ≢ −1 (mod N)`, then from
  `(x^{r/2} − 1)(x^{r/2} + 1) = x^r − 1 ≡ 0 (mod N)`
  with `N` dividing neither factor, `gcd(x^{r/2} − 1, N)` is a nontrivial divisor.

**Success of the reduction.** With `N = ∏_{i=1}^k p_i^{a_i}` (`k` distinct odd primes), it fails only if the 2-adic valuations of the orders `r_i = ord_{p_i^{a_i}}(x)` all agree: all zero gives odd `r`; a common positive value gives `x^{r/2} ≡ −1` modulo every prime-power factor. Each `(Z/p_i^{a_i})*` is cyclic, so each valuation takes any fixed value with probability `≤ 1/2`; by CRT the `k` choices are independent, so they all agree with probability `≤ 1/2^{k−1}`. Hence a random `x` yields a factor with probability `≥ 1 − 1/2^{k−1} ≥ 1/2`.

## Quantum order-finding

Let `q = 2^m` with `N² ≤ q < 2N²`. Two registers.

1. Uniform superposition (Hadamard each input qubit):  `q^{-1/2} Σ_{a=0}^{q−1} |a⟩|0⟩`.
2. Reversible modular exponentiation:  `q^{-1/2} Σ_{a} |a⟩|x^a mod N⟩`.
3. Quantum Fourier transform on register 1, `|a⟩ → q^{-1/2} Σ_c e^{2πi ac/q}|c⟩`:
   `(1/q) Σ_{a,c} e^{2πi ac/q} |c⟩|x^a mod N⟩`.
4. Measure. Probability of `|c, x^k mod N⟩` (sum over `a ≡ k mod r`, `a = br + k`):
   `P_k(c) = | (1/q) Σ_{b=0}^{⌊(q−k−1)/r⌋} e^{2πi b{rc}_q/q} |²`,
   where `{rc}_q ∈ (−q/2, q/2]` is `rc mod q`.

**Constructive interference.** If `|{rc}_q| ≤ r/2`, compare the geometric sum with an integral and substitute `u = rb/q`:
`(1/r) ∫_0^1 e^{2πi ({rc}_q/r) u} du`, minimized in modulus at `{rc}_q/r = ±1/2` giving `2/(πr)`. The sum differs by `O(1/q)`, so the asymptotic lower bound is `4/(π² r²)`; because `4/π² > 1/3`, the usable bound is `P_k(c) ≥ 1/(3r²)` for sufficiently large `N`.

## Continued-fraction recovery

`|{rc}_q| ≤ r/2` ⇔ `∃ d: |c/q − d/r| ≤ 1/(2q)`. Because `q ≥ N²` and `r < N`, two distinct fractions with denominator `< N` differ by `> 1/N² ≥ 1/q`, so `d/r` is the **unique** such fraction near `c/q`; the continued-fraction expansion of `c/q` (convergents `p_n/q_n` via `p_n = a_n p_{n−1} + p_{n−2}`, `q_n = a_n q_{n−1} + q_{n−2}`) finds it in polynomial time. The reduced denominator is `r` exactly when `gcd(d, r) = 1`.

## Overall success probability

There are `φ(r)` values of `d` coprime to `r` (each giving one good `c`) and `r` values of `x^k`, so `r·φ(r)` favorable outcomes, each with probability `≥ 1/(3r²)`:
`P(recover r) ≥ φ(r)/(3r)`.
Since `φ(r)/r > δ/log log r` (Hardy–Wright Thm 328), each run succeeds with probability `≳ 1/log log r`, so `O(log log r)` repetitions suffice.

## Quantum Fourier transform circuit (correctness)

For the QFT paragraph, write `q = 2^m`. Gates: Hadamard `R_j` on bit `j`; controlled phase `S_{j,k}` (`j<k`) applying `e^{iθ}`, `θ_{k−j} = π/2^{k−j}`, on `|11⟩`. Apply `R_{m−1} S_{m−2,m−1} R_{m−2} … R_0` — `m` Hadamards plus `m(m−1)/2` controlled phases. The amplitude `|a⟩→|b⟩` picks up phase
`Σ_{0≤j<m} π a_j b_j + Σ_{0≤j<k<m} (π/2^{k−j}) a_j b_k = Σ_{0≤j≤k<m} (π/2^{k−j}) a_j b_k`.
With `b` the bit-reversal of `c` (`b_k = c_{m−1−k}`), reindexing `k ← m−1−k` gives
`Σ_{j+k<m} 2π (2^j 2^k/2^m) a_j c_k`. Terms with `j+k ≥ m` are integer multiples of `2π`, so the phase may be extended to
`Σ_{j,k=0}^{m−1} 2π (2^j 2^k/2^m) a_j c_k = 2π ac/q`,
i.e. exactly `e^{2πi ac/q}` with `1/√q` from the Hadamards. Coppersmith's approximate QFT drops the exponentially small phases (large `k−j`) and still factors.

## Cost

Modular exponentiation dominates for input length `l = log N`: `O(l³)` time, `O(l)` space (longhand multiplication), or `O(l² log l log log l)` with Schönhage–Strassen. The QFT costs `O((log q)^2) = O(l²)`. Times `O(log log r)` repetitions, plus polynomial classical post-processing (gcd, continued fractions). Total polynomial in `l`.

## Implementation (period-finder + Miller reduction, classically simulated for small N)

```python
import math, random
from collections import defaultdict
from fractions import Fraction

def gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def modexp(base, e, mod):
    r = 1; base %= mod
    while e:
        if e & 1: r = (r * base) % mod
        base = (base * base) % mod; e >>= 1
    return r

def _integer_nth_root(n, k):
    lo, hi = 1, 1 << ((n.bit_length() + k - 1) // k)
    while lo <= hi:
        mid = (lo + hi) // 2
        p = mid ** k
        if p == n:
            return mid
        if p < n:
            lo = mid + 1
        else:
            hi = mid - 1
    return hi

def _perfect_power_factor(N):
    for k in range(2, N.bit_length() + 1):
        root = _integer_nth_root(N, k)
        if root > 1 and root ** k == N:
            return root
    return None

def find_period(x, N):
    q_bits = (N * N - 1).bit_length()     # q = 2^q_bits, N^2 <= q < 2 N^2
    q = 1 << q_bits
    # exact simulation of: superposition -> modexp -> QFT -> measure
    groups = defaultdict(list)
    value = 1
    for a in range(q):
        groups[value].append(a)
        value = (value * x) % N
    probs = [0.0] * q
    for alist in groups.values():
        for c in range(q):
            s = sum(complex(math.cos(2*math.pi*a*c/q),
                            math.sin(2*math.pi*a*c/q)) for a in alist)
            probs[c] += abs(s / q) ** 2
    t = random.random(); acc = 0.0
    c = q - 1
    for cc in range(q):
        acc += probs[cc]
        if t <= acc: c = cc; break
    frac = Fraction(c, q).limit_denominator(N - 1)
    candidate = frac.denominator
    if candidate > 0 and modexp(x, candidate, N) == 1:
        return candidate
    return None

def factor(N):
    if N < 2:
        raise ValueError("N must be at least 2")
    if N % 2 == 0: return 2
    pp = _perfect_power_factor(N)
    if pp is not None:
        return pp
    while True:
        x = random.randrange(2, N)
        g = gcd(x, N)
        if g != 1:
            return g                       # lucky common factor
        r = find_period(x, N)
        if r is None or r % 2 != 0:
            continue                       # need even order
        y = modexp(x, r // 2, N)
        if y == 1 or y == N - 1:
            continue                       # retry the failed reduction cases
        f = gcd(y - 1, N)
        if 1 < f < N:
            return f                       # gcd(x^{r/2}-1, N)
```

Worked instance: `N = 91 = 7·13`, `x = 3` has order `r = 6`; `3^3 = 27 ≢ −1 (mod 91)`, and `gcd(27 − 1, 91) = gcd(26, 91) = 13`.
