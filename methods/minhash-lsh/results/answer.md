# MinHash and Locality-Sensitive Hashing

## Problem

Given an enormous collection of objects — billions of web pages, each turned into a *set* of features (w-shingles: every contiguous run of w tokens) — solve two coupled problems cheaply:

1. **Estimate set similarity.** For two sets A, B, recover the Jaccard resemblance J(A,B) = |A∩B|/|A∪B| without scanning the full sets. Exact J costs O(|A|+|B|) per pair and requires storing both full sets.
2. **Find the near-duplicates / nearest neighbors.** Across N objects, return the pairs (or, per query, the items) with similarity above a threshold — without the brute-force Θ(N²) all-pairs comparison, and in high dimension where space-partitioning trees collapse to a linear scan (the curse of dimensionality).

A few percent estimation error and a small, tunable false-positive / false-negative rate are acceptable; that slack is what buys the sublinearity.

## Key idea

**MinHash.** Hash every element to a value treated as uniform in a large range (this makes the statistic a function of the *set*, not the multiset, and ties the randomness to the element so two sketches are coordinated). For a random permutation π of the universe, define the minhash h(S) = the element of S with the smallest hash. The element of A∪B with the global smallest hash is uniform over A∪B, and the two sets' minima agree exactly when that element lies in A∩B:

    Pr[h(A) = h(B)] = |A∩B| / |A∪B| = J(A,B).

The collision probability *equals* the target — not approximately. One minhash is a Bernoulli(J); averaging k independent ones gives an unbiased estimate Ĵ = (1/k)·#{i : hᵢ(A)=hᵢ(B)} with variance J(1−J)/k, so standard error O(1/√k) and k = O(1/ε²) hashes for error ε. The signature is k small integers regardless of set size, computed in one streaming pass, and merges under set union by elementwise minimum.

**LSH banding (all-pairs near-duplicate detection).** Comparing all N²/2 signatures is still quadratic. Split the length-k signature into b bands of r rows (usually k = b·r; an implementation may use b·r ≤ k and leave unused tail slots). Two sets are a candidate pair if they agree on *all* r rows of *at least one* band — AND within a band, OR across bands. With a single row agreeing with probability s = J:

    Pr[candidate] = 1 − (1 − sʳ)ᵇ.

This is an S-curve: nearly flat and low for small s, a steep rise through a threshold. The exact P(candidate)=1/2 point is (1−2^(-1/b))^(1/r); the standard design rule uses the approximation s* ≈ (1/b)^(1/r). The AND (sʳ) drives moderate similarities toward 0; the OR (1−(1−·)ᵇ) restores recall for high similarities. Hash each band into its own hash table; building b tables is one linear pass over the N signatures, and a query inspects only its b buckets, so only the candidate list gets the exact O(k) verification.

**LSH for sublinear ANN (per-query nearest neighbor).** Abstract the minhash to a *locality-sensitive family*: H is (r₁, r₂, p₁, p₂)-sensitive if d(x,y) ≤ r₁ ⇒ Pr[h(x)=h(y)] ≥ p₁ and d(x,y) ≥ r₂ ⇒ Pr[h(x)=h(y)] ≤ p₂, with p₁ > p₂. (MinHash is exactly this for Jaccard distance d = 1−J: a single minhash agrees with probability s = 1−d.) Build L tables, each keyed by gⱼ(x) = (h₁(x),…,h_k(x)) — concatenation of k base functions is the AND that sharpens. Choosing k = log N / log(1/p₂) makes a far point's per-table collision probability p₂ᵏ ≈ 1/N; a near point survives one table with probability ≥ p₁ᵏ = N^{−ρ}, where

    ρ = log(1/p₁) / log(1/p₂),   0 < ρ < 1.

Taking L = O(N^ρ) tables catches the near point with constant probability while the expected far-point collisions across the queried buckets stay O(L). Query cost is O(N^ρ) hash evaluations and distance checks, space O(d·N + N^{1+ρ}) — sublinear. For the bit-sampling family over {0,1}^d under Hamming distance, the log-ratio exponent is bounded by ρ ≤ 1/c for approximation factor c, which gives the standard n^(1/c) query bound; a 2-approximate near-neighbor query costs ≈ √N instead of N.

The two halves lock together: the same minhash that gives an unbiased Jaccard estimate *is* a locality-sensitive function for Jaccard distance, and banding/concatenation is the AND-then-OR amplification that both generates a short candidate list (b-bands form) and answers single queries in N^ρ time (L-tables form).

## Algorithm

MinHash signature (one pass, mergeable):
- Initialize k slots to +∞.
- For each element v of the set and each i ∈ {1,…,k}: slotᵢ ← min(slotᵢ, (aᵢ·hash(v) + bᵢ) mod p), reduced into a 32-bit range, with p a Mersenne prime (2⁶¹−1) and random aᵢ ∈ [1,p), bᵢ ∈ [0,p). The k maps a·x+b mod p are cheap pairwise-independent stand-ins for k random permutations; a true permutation of billions of rows is unstorable, and finite-range collisions are treated as rare implementation noise.
- Ĵ(A,B) = fraction of agreeing slots.

LSH index:
- Pick (b, r) with b·r ≤ k placing the S-curve threshold ≈ (1/b)^(1/r) at the target resemblance; the implementation choice below minimizes a weighted sum of the false-positive area ∫₀ᵗ (1−(1−sʳ)ᵇ) ds and false-negative area ∫ₜ¹ (1−sʳ)ᵇ ds.
- Insert: hash each of the b bands (an r-tuple of slots) into its own table; N·b bucket insertions in one linear pass.
- Query: collect the ids colliding in any band's bucket; verify candidates exactly.

## Code

```python
import numpy as np
import hashlib
import struct
from scipy.integrate import quad as integrate

MERSENNE_PRIME = np.uint64((1 << 61) - 1)   # p for the a*x+b mod p maps
MAX_HASH = np.uint64((1 << 32) - 1)         # reduce into a 32-bit range
HASH_RANGE = 1 << 32

def sha1_hash32(data):
    # fixed pseudo-uniform hash: bytes -> {0,...,2^32-1}
    return struct.unpack("<I", hashlib.sha1(data).digest()[:4])[0]


class MinHash:
    def __init__(self, num_perm=128, seed=1, hashfunc=sha1_hash32):
        if num_perm > HASH_RANGE:
            raise ValueError("Cannot have more than 2^32 permutation functions")
        self.seed = seed
        self.num_perm = num_perm
        self.hashfunc = hashfunc
        self.hashvalues = self._init_hashvalues(num_perm)
        self.permutations = self._init_permutations(num_perm)

    def _init_hashvalues(self, num_perm):
        return np.ones(num_perm, dtype=np.uint64) * MAX_HASH

    def _init_permutations(self, num_perm):
        gen = np.random.RandomState(self.seed)
        # k cheap pairwise-independent maps standing in for random permutations
        return np.array(
            [
                (
                    gen.randint(1, MERSENNE_PRIME, dtype=np.uint64),
                    gen.randint(0, MERSENNE_PRIME, dtype=np.uint64),
                )
                for _ in range(num_perm)
            ],
            dtype=np.uint64,
        ).T

    def update(self, value):
        hv = self.hashfunc(value)
        a, b = self.permutations
        # the k "permuted" images of this element, reduced to 32 bits
        phv = np.bitwise_and((a * hv + b) % MERSENNE_PRIME, MAX_HASH)
        # each slot records the MINIMUM image seen across the set's elements
        self.hashvalues = np.minimum(phv, self.hashvalues)

    def merge(self, other):
        if self.seed != other.seed or len(self) != len(other):
            raise ValueError("Cannot merge MinHash sketches with different parameters")
        # union of two sets == elementwise min of their minhash signatures
        self.hashvalues = np.minimum(self.hashvalues, other.hashvalues)

    def jaccard(self, other):
        if self.seed != other.seed or len(self) != len(other):
            raise ValueError("Cannot compare MinHash sketches with different parameters")
        # fraction of agreeing coordinates is an unbiased estimate of J(A,B)
        return float(np.count_nonzero(self.hashvalues == other.hashvalues)) / float(len(self))

    def __len__(self):
        return len(self.hashvalues)


def _false_positive_probability(threshold, b, r):
    # candidate mass below threshold: 1 - (1 - s^r)^b
    return integrate(lambda s: 1 - (1 - s ** float(r)) ** float(b), 0.0, threshold)[0]

def _false_negative_probability(threshold, b, r):
    # miss mass above threshold: 1 - candidate = (1 - s^r)^b
    return integrate(lambda s: 1 - (1 - (1 - s ** float(r)) ** float(b)), threshold, 1.0)[0]

def _optimal_param(threshold, num_perm, false_positive_weight, false_negative_weight):
    # grid-search b, r (b*r <= num_perm) by weighted FP/FN area
    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = _false_positive_probability(threshold, b, r)
            fn = _false_negative_probability(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


class MinHashLSH:
    def __init__(self, threshold=0.9, num_perm=128, weights=(0.5, 0.5), params=None, hashfunc=None):
        if threshold > 1.0 or threshold < 0.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        if num_perm < 2:
            raise ValueError("Too few permutation functions")
        if any(w < 0.0 or w > 1.0 for w in weights) or sum(weights) != 1.0:
            raise ValueError("weights must be nonnegative and sum to 1.0")
        self.h = num_perm
        # AND within a band (r rows), OR across bands (b bands): 1-(1-s^r)^b S-curve
        if params is None:
            self.b, self.r = _optimal_param(threshold, num_perm, weights[0], weights[1])
        else:
            self.b, self.r = params
            if self.b * self.r > num_perm:
                raise ValueError("b * r must be less than or equal to num_perm")
        if self.b < 2:
            raise ValueError("The number of bands is too small")
        self.hashranges = [(i * self.r, (i + 1) * self.r) for i in range(self.b)]
        self.hashtables = [dict() for _ in range(self.b)]
        self.keys = {}
        self.hashfunc = hashfunc
        self._H = self._hashed_byteswap if hashfunc else self._byteswap

    def _byteswap(self, hs):
        return bytes(hs.byteswap().data)

    def _hashed_byteswap(self, hs):
        return self.hashfunc(bytes(hs.byteswap().data))

    def insert(self, key, m):
        if len(m) != self.h:
            raise ValueError("Expecting MinHash with the configured length")
        # each band is the r-tuple of minhashes in that slice, serialized byteswapped
        hs = [self._H(m.hashvalues[start:end]) for start, end in self.hashranges]
        self.keys[key] = hs
        for h, table in zip(hs, self.hashtables):       # one linear pass: N*b insertions
            table.setdefault(h, set()).add(key)

    def query(self, m):
        if len(m) != self.h:
            raise ValueError("Expecting MinHash with the configured length")
        # candidates = everything colliding in ANY band; caller verifies exactly
        candidates = set()
        for (start, end), table in zip(self.hashranges, self.hashtables):
            candidates.update(table.get(self._H(m.hashvalues[start:end]), ()))
        return list(candidates)
```
