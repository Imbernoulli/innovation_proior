# AMS frequency-moment estimation: the Tug-of-War sketch for F_2

## Problem

A stream `a_1, ..., a_m` of values from `[n]` defines frequencies `m_i = #{j : a_j = i}`. The `k`-th frequency moment is `F_k = sum_i m_i^k`. We want a one-pass, small-space, randomized `(λ, δ)`-approximation: output `Y` with `P(|Y - F_k| > λ F_k) ≤ δ`. `F_2 = sum_i m_i^2` (self-join size / repeat rate) is the headline case. An exact histogram needs `Θ(n log m)` bits; we want polylogarithmic space.

## Key idea (F_2): a randomized linear sketch, squared

Maintain a single scalar that is a *linear* function of the frequencies, with random `±1` coefficients, and square it at the end:

- Draw signs `ε_i = h(i) ∈ {-1, +1}` from a **four-wise independent** family (seed `O(log n)` bits).
- Keep `Z = sum_i ε_i m_i`; update `Z += ε_{a_j}` per token. `|Z| ≤ m`, so `O(log m)` bits.
- Estimator `X = Z^2`.

**Unbiased.** `Z^2 = sum_i ε_i^2 m_i^2 + sum_{i≠j} ε_i ε_j m_i m_j`. With `ε_i^2 = 1` and mean-zero pairwise-independent signs, `E[ε_i ε_j] = 0` for `i ≠ j`, so `E[Z^2] = sum_i m_i^2 = F_2`.

**Variance.** Writing `S = sum_{i<j} m_i^2 m_j^2`: four-wise independence kills every fourth-moment term with a "lonely" index, leaving `E[Z^4] = F_4 + 6S`. Since `F_2^2 = F_4 + 2S`,

```
Var(Z^2) = E[Z^4] - F_2^2 = (F_4 + 6S) - (F_4 + 2S) = 4S = 2(2S) ≤ 2 F_2^2,
```

using `2S ≤ F_4 + 2S = F_2^2`. Four-wise independence is exactly what the variance proof needs, and small four-wise sign families (a degree-3 polynomial over `GF(2^d)` followed by one output bit, or a BCH / strength-4 orthogonal array) cost only an `O(log n)`-bit seed.

## Final algorithm (F_2): median-of-means

A single `X` has relative std `√2`, too large. Amplify in two stages:

- Average `s_1 = 16/λ^2` independent copies: `Y = (1/s_1) sum_t (Z^{(t)})^2`, `Var(Y) ≤ 2F_2^2/s_1`. Chebyshev gives `P(|Y - F_2| > λ F_2) ≤ 2/(s_1 λ^2) ≤ 1/8`.
- Take the **median** of `s_2 = 2 log_2(1/δ)` such averages. The median is bad only if at least half the row averages are bad, and `2^{s_2}(1/8)^{s_2/2} = 2^{-s_2/2} ≤ δ`.

Total space `O(λ^{-2} log(δ^{-1}) (log n + log m))` bits — logarithmic in `n` and `m`.

## General F_k (k ≥ 2)

Pick a uniform stream position `p` (reservoir sampling), let `l = a_p`, `r = #{q ≥ p : a_q = l}`, and set `X = m(r^k - (r-1)^k)`. Telescoping gives `E[X] = F_k`. Using `c^k - (c-1)^k ≤ k c^{k-1}`, `E[X^2] ≤ k F_1 F_{2k-1} ≤ k n^{1-1/k} F_k^2`. Median-of-means then yields `O(k λ^{-2} log(δ^{-1}) n^{1-1/k}(log n + log m))` bits. For `k = 2` this is `√n`; the tug-of-war sketch is the special improvement to `log n`.

## F_0 with an honest hash

Linear hash `z = a·x + b` over `GF(2^d)`, `2^d > n`. Track `R = max` trailing-zero count `r(z)` over the stream; output `2^R`. Pairwise independence suffices to show `2^R` is within a constant factor of `F_0` with constant probability, in `O(log n)` bits.

## Lower bounds

Via reductions to (multiparty) Disjointness in communication complexity: approximating `F_∞* = max_i m_i` needs `Ω(n)`. For `F_k`, the `DIS(s,t)` game needs `Ω(t/s^3)` communication; a streaming algorithm with `M` bits gives an `sM`-bit protocol, so `M = Ω(t/s^4)`. Choosing `s = Θ(n^{1/k})` and `t = Θ(n^{1-1/k})` gives `Ω(n^{1-5/k})` for `k > 5`, even with a constant number of passes. Deterministic or exact estimation of `F_k` (`k ≠ 1`) needs `Ω(n)` — so randomness and approximation are both necessary. Matching floors: `Ω(log n)` for `F_0`, `Ω(log log m)` for `F_1`, `Ω(log n + log log m)` for `F_2`.

## Code (F_2 Tug-of-War sketch)

```python
import math, random, statistics

FIELD_BITS = 64
FIELD_MASK = (1 << FIELD_BITS) - 1
REDUCTION = 0x1B  # x^64 + x^4 + x^3 + x + 1


def gf_mul(a, b):
    z = 0
    a &= FIELD_MASK
    b &= FIELD_MASK
    while b:
        if b & 1:
            z ^= a
        b >>= 1
        carry = a >> (FIELD_BITS - 1)
        a = (a << 1) & FIELD_MASK
        if carry:
            a ^= REDUCTION
    return z


class StreamHash:
    """epsilon_i = +/-1 from a degree-3 polynomial over GF(2^64)."""
    def __init__(self, rng):
        self.a = [rng.getrandbits(FIELD_BITS) for _ in range(4)]

    def __call__(self, x):
        if not 0 <= x <= FIELD_MASK:
            raise ValueError("token is outside the represented universe")
        a0, a1, a2, a3 = self.a
        v = gf_mul(a3, x) ^ a2
        v = gf_mul(v, x) ^ a1
        v = gf_mul(v, x) ^ a0
        return 1 if (v & 1) == 0 else -1


class StreamSummary:
    def __init__(self, eps, delta, seed=0):
        if eps <= 0:
            raise ValueError("eps must be positive")
        if not 0 < delta < 1:
            raise ValueError("delta must be in (0, 1)")
        rng = random.Random(seed)
        self.s1 = max(1, math.ceil(16 / (eps * eps)))         # average: 16/lambda^2
        self.s2 = max(1, math.ceil(2 * math.log2(1 / delta))) # median: 2 log_2(1/delta)
        self.hashes = [[StreamHash(rng) for _ in range(self.s1)] for _ in range(self.s2)]
        self.z = [[0] * self.s1 for _ in range(self.s2)]

    def update(self, token, count=1):
        for r in range(self.s2):
            for c in range(self.s1):
                self.z[r][c] += count * self.hashes[r][c](token)

    def estimate(self):
        values = [z * z for row in self.z for z in row]
        return median_of_means(values, self.s1, self.s2)


def median_of_means(per_copy_values, s1, s2):
    rows = [per_copy_values[r * s1:(r + 1) * s1] for r in range(s2)]
    return statistics.median(sum(row) / s1 for row in rows)
```
