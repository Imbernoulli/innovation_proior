# HyperLogLog — cardinality estimation in sublinear space

## Problem

Estimate the number of distinct elements n in a read-once stream of arbitrary size, in a single pass, using a small fixed amount of memory (a few hundred bytes to a couple of kilobytes), with the answer insensitive to how often each value repeats. Exact counting needs Θ(n) memory; HyperLogLog gives a relative error of about 1.04/√m using m small registers.

## Key idea

Hash every element to uniform bits, so any summary computed from the hashes depends only on the *set* of values (repeats land on themselves) and supplies the randomness the analysis needs. The observable is the **leading-zero run**: for a hashed string, ρ = position of the leftmost 1-bit = 1 + length of the run of leading zeros. Among n uniform strings the largest ρ sits within an additive constant of log₂ n, so a long leading-zero run is a thermometer for cardinality.

A single observable has standard deviation over a full bit, so combine m of them by **stochastic averaging**: use the first p bits of the hash to send each element to one of m = 2^p buckets, and the remaining bits to compute ρ. Each bucket keeps only the **maximum ρ** it has seen, a register M[j] ≈ log₂(n/m) needing only ~log₂ log₂ n bits (5-bit "short bytes").

Combine the registers with the **harmonic mean** of the 2^{M[j]} (not the arithmetic or geometric mean). The 2^{M[j]} have a slowly decaying right tail — an occasional freak-long leading-zero run doubles or quadruples a bucket's value — and the harmonic mean weights each term by its reciprocal, so those freak-large buckets contribute almost nothing and cannot inflate the estimate. This is the single change from LogLog (which uses the geometric mean) and it drops the standard error from 1.30/√m to 1.04/√m, close to the ~1/√m benchmark set by the order-statistics family.

## Algorithm

Registers M[1..m] initialized to 0, m = 2^p. For each element v:
- x = h(v); j = first p bits of x; w = remaining bits; M[j] = max(M[j], ρ(w)).

Estimate, with indicator Z = (Σ_{j=1}^m 2^{-M[j]})^{-1}:

  E = α_m · m² · Z = α_m · m² / Σ_{j=1}^m 2^{-M[j]},

where the bias-correction constant is

  α_m = ( m ∫_0^∞ (log₂((2+u)/(1+u)))^m du )^{-1},  α_∞ = 1/(2 ln 2) ≈ 0.72134,

evaluated for small m as α_16 = 0.673, α_32 = 0.697, α_64 = 0.709, and α_m = 0.7213/(1 + 1.079/m) for m ≥ 128.

Range corrections (32-bit hash):
- **Small range**, E ≤ 5m/2: let V = number of registers equal to 0. If V ≠ 0, return linear counting E* = m·log(m/V) (empty bins ≈ m·e^{−n/m}); otherwise return E.
- **Middle range**, 5m/2 < E ≤ 2³²/30: return E.
- **Large range**, E > 2³²/30: hash collisions saturate, so, while E < 2³², invert E = 2³²(1 − e^{−n/2³²}) to E* = −2³²·log(1 − E/2³²). If E reaches 2³², a 32-bit hash has no finite collision inversion left.

Standard error ≈ 1.04/√m; estimates lie within σ, 2σ, 3σ of the truth in about 65%, 95%, 99% of cases. With m = 2048, 5-bit packed registers take about 1.3 kB and give ~2% error up to 10⁹. The register size matches the Ω(log log N) information scale, and the accuracy is near the known ~1/√m comparison point.

## Code

```python
import math
from hashlib import sha1

HASH_BITS = 32
HASH_RANGE = 1 << HASH_BITS

def hash32(value):
    data = value if isinstance(value, bytes) else str(value).encode("utf-8")
    return int.from_bytes(sha1(data).digest()[:4], byteorder="big")

def rho(w, max_width):
    # leftmost 1-bit position = 1 + leading-zero run within max_width bits
    # rho(all zero bits) = max_width + 1
    return max_width - w.bit_length() + 1

def alpha(m):
    if m == 16:  return 0.673
    if m == 32:  return 0.697
    if m == 64:  return 0.709
    return 0.7213 / (1.0 + 1.079 / m)          # m >= 128

class HyperLogLog:
    def __init__(self, p=11):
        # m = 2**p registers; standard error is about 1.04 / sqrt(m)
        if not (4 <= p <= 16):
            raise ValueError("32-bit hash range requires 4 <= p <= 16")
        self.p = p
        self.m = 1 << p
        self.alpha = alpha(self.m)
        self.M = [0] * self.m

    def add(self, value):
        x = hash32(value)
        j = self._bucket_index(x)
        w = self._remaining_bits(x)
        obs = self._observable(w)
        self.M[j] = self._combine_into_bucket(self.M[j], obs)

    def merge(self, other):
        if self.p != other.p:
            raise ValueError("precisions must match")
        self.M = [
            self._combine_buckets(a, b)
            for a, b in zip(self.M, other.M)
        ]

    def _bucket_index(self, x):
        suffix_width = HASH_BITS - self.p
        return x >> suffix_width                    # first p bits -> bucket

    def _remaining_bits(self, x):
        suffix_width = HASH_BITS - self.p
        return x & ((1 << suffix_width) - 1)        # remaining bits -> observable

    def _observable(self, w):
        return rho(w, HASH_BITS - self.p)

    def _combine_into_bucket(self, current, obs):
        return max(current, obs)                    # keep the max rho per bucket

    def _combine_buckets(self, left, right):
        return max(left, right)

    def estimate(self):
        m = self.m
        Z = 1.0 / sum(2.0 ** -mj for mj in self.M)   # harmonic mean via indicator
        E = self.alpha * m * m * Z

        if E <= 2.5 * m:                              # small-range correction
            V = self.M.count(0)
            if V != 0:
                return m * math.log(m / V)            # linear counting
            return E
        if E <= HASH_RANGE / 30.0:                    # middle band
            return E
        if E >= HASH_RANGE:
            return float("inf")                       # saturated 32-bit hash range
        return -HASH_RANGE * math.log1p(-E / HASH_RANGE)  # large-range correction

    def count(self):
        return self.estimate()
```

The merge operation (register-wise max) makes HyperLogLog distributable and parallelizable with no loss of accuracy: counters built on disjoint shards combine into the counter for the union, since the max leading-zero run over a union is the max of the per-shard maxima.
