We need to estimate the number of distinct elements in a read-once stream that may contain billions of records, using only a small, fixed amount of memory and a few operations per element. Exact counting is impossible under these constraints because certifying that a newly seen value is distinct requires remembering every distinct value encountered so far, which costs Θ(n) space. Sampling is equally unsuitable: a random sample estimates the number of records weighted by their frequencies, not the number of distinct values, and the estimate explodes or collapses depending on the skew of the repetition pattern. The statistic we want must depend only on the set of values, not on how often each one appears.

The escape route is to hash every element to a uniform random binary string. Identical records collide to the same hash, so any function of the collection of hashes is automatically a function of the set of distinct hashes and therefore replication-insensitive. At the same time, hashing supplies the uniform randomness we need. The key observation is that rare bit patterns are evidence of large cardinality: a uniform binary string starts with k zeros followed by a one with probability 2^{-(k+1)}, so seeing a long leading-zero run suggests that many distinct hashes have been examined. The position ρ of the leftmost 1-bit is a thermometer for log₂ n. A single such measurement is far too noisy, with standard deviation over a full bit, so we need many independent measurements and a way to combine them cheaply.

The method is HyperLogLog. It splits the hash into two parts. The first p bits of the hash send each element to one of m = 2^p buckets; the remaining bits are used to compute ρ, the one-based position of the leftmost 1-bit. Each bucket stores only the maximum ρ it has ever seen, a small register M[j] ≈ log₂(n/m). Because the register value itself is on a logarithmic scale, storing it takes only about log₂ log₂ n bits, so the whole sketch collapses from a bitmap per bucket to a "short byte" per bucket. The crucial variance-reduction step is how the registers are combined. Earlier LogLog estimators average the registers in the exponent, which is equivalent to taking the geometric mean of the 2^{M[j]}. That leaves accuracy on the table because the distribution of 2^{M[j]} has a slowly decaying right tail: an occasional freakishly long leading-zero run in one bucket doubles or quadruples that bucket's contribution and inflates the estimate. HyperLogLog instead uses the harmonic mean of the 2^{M[j]}. The harmonic mean weights each term by its reciprocal, so a bucket with an outlandishly large value contributes almost nothing and cannot pull the estimate up. This single change drops the standard error from 1.30/√m to about 1.04/√m, close to the ~1/√m accuracy barrier.

Concretely, define the indicator Z = (Σ_j 2^{-M[j]})^{-1}. The raw estimate is E = α_m · m² · Z, where α_m is a bias-correction constant derived from the Poissonized analysis. In the limit α_∞ = 1/(2 ln 2) ≈ 0.72134; for small m the constants are tabulated, and for m ≥ 128 the approximation α_m = 0.7213/(1 + 1.079/m) is accurate. Two practical corrections keep the estimate honest across the whole range. When E ≤ 5m/2 and some registers are still zero, the empty-bucket fraction is a better signal than the harmonic formula, so we switch to linear counting: E* = m · log(m/V), where V is the number of zero registers. When E approaches the 2³² hash range, distinct hashes begin colliding and the raw estimate undercounts; while E < 2³² we invert the collision relation to get E* = −2³² · log(1 − E/2³²). Between these extremes the raw estimate is used directly.

```python
import math
from hashlib import sha1

HASH_BITS = 32
HASH_RANGE = 1 << HASH_BITS

def hash32(value):
    data = value if isinstance(value, bytes) else str(value).encode("utf-8")
    return int.from_bytes(sha1(data).digest()[:4], byteorder="big")

def rho(w, max_width):
    if not (0 <= w < (1 << max_width)):
        raise ValueError("w does not fit in max_width bits")
    return max_width - w.bit_length() + 1

def alpha(m):
    if m == 16:  return 0.673
    if m == 32:  return 0.697
    if m == 64:  return 0.709
    return 0.7213 / (1.0 + 1.079 / m)

class HyperLogLog:
    def __init__(self, p=11):
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
        self.M[j] = max(self.M[j], rho(w, HASH_BITS - self.p))

    def merge(self, other):
        if self.p != other.p:
            raise ValueError("precisions must match")
        self.M = [max(a, b) for a, b in zip(self.M, other.M)]

    def _bucket_index(self, x):
        return x >> (HASH_BITS - self.p)

    def _remaining_bits(self, x):
        return x & ((1 << (HASH_BITS - self.p)) - 1)

    def estimate(self):
        m = self.m
        Z = 1.0 / sum(2.0 ** -mj for mj in self.M)
        E = self.alpha * m * m * Z

        if E <= 2.5 * m:
            V = self.M.count(0)
            if V != 0:
                return m * math.log(m / V)
            return E
        if E <= HASH_RANGE / 30.0:
            return E
        if E >= HASH_RANGE:
            return float("inf")
        return -HASH_RANGE * math.log1p(-E / HASH_RANGE)

    def count(self):
        return self.estimate()
```
