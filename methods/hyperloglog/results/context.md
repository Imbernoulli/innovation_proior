# Context — Counting distinct elements of a massive stream in tiny memory

## Research question

Given a very large multiset M — a stream of records read once, left to right, possibly billions of items, with arbitrary repetition — estimate the **cardinality** n, the number of *distinct* elements, using a small, fixed amount of auxiliary memory and only a few operations per item. The constraints that make this hard are concrete:

- **One pass, read-once.** The data is a stream (network packets, log lines, a file too big for core). We cannot sort, cannot revisit, cannot hold the data.
- **Sub-linear memory, fixed in advance.** Anything proportional to n is unacceptable; n itself is what we are trying to learn and may be 10^9. For a chosen accuracy, we want the stored representation of each maintained value to grow like a logarithm of a logarithm of the range, so the whole sketch stays in a few hundred bytes to a couple of kilobytes.
- **Insensitivity to replication.** Whether a value occurs once or a million times, it must count exactly once. The answer must depend only on the *set* of values present, not on their frequencies.
- **A few percent error is acceptable.** This is the relaxation that makes everything possible: we estimate n approximately, not exactly.

Why it matters: query optimizers in databases choose join orders and access paths from the cardinalities of key fields; network monitors detect worm propagation and denial-of-service by watching the number of distinct active flows (a worm opens many *different* connections, invisible in raw traffic volume but loud in cardinality); data-mining and graph systems repeatedly need distinct counts of huge sets. In all of these the cardinality is queried far more often, and on far larger data, than an exact method can afford.

## Background

**Why exact counting is linear.** To know that an element is distinct, you must remember every distinct element you have already seen — a hash table or a sorted, deduplicated list. That is Θ(n) words of storage and Θ(n) or Θ(n log n) work. For n in the billions this is exactly the cost we are trying to avoid; it defeats the purpose of having a cheap statistic.

**Why sampling fails for cardinality.** The obvious shortcut — keep a random sample of size N₀ from a stream of N records, count distinct values v₀ in the sample, return v₀·(N/N₀) — estimates the number of *records*, not distinct values, and is badly biased under skew. If one value dominates the occurrences, the sample is full of it and sees few distinct values; if the distinct values are spread thin over many records, the scaled estimate explodes. Cardinality is sensitive to the replicative structure of the data in a way that frequency-based sampling cannot track.

**Hashing converts data to randomness.** All efficient cardinality estimators rest on a single modeling device: apply a fixed hash function h: D → {0,1}^L whose output bits are, to good approximation, independent and each 1 with probability ½. Equivalently the hashed values are uniform reals in [0,1]. Standard hashing (multiplicative, CRC-based, a cryptographic primitive used loosely) achieves this in practice on real files. Hashing buys two things at once: it manufactures the uniform randomness the analysis needs, and — because identical records hash identically — it makes every observable depend only on the *set* of hashed values, automatically insensitive to replication. This is the structural reason these methods are idempotent under repeats.

**The log log n target was already in the air.** Morris's approximate counting (1977) maintains an approximate counter that can count up to n using about log₂ log₂ n + O(1) bits, by storing an exponent and incrementing it only probabilistically. That counts *occurrences*, not distinct values, but it establishes that a count up to n can live in doubly-logarithmic memory — a quantity of the right order is being maintained, not the count itself but its logarithm's logarithm. It sets the memory bar a distinct-counter should aspire to.

**Observables that depend only on the set.** Define an *observable* of the multiset of hashed values to be any function of the *set* underlying it. Two families have been studied:

- **Bit-pattern observables.** Look at patterns at the beginning of the hashed strings. If among n uniform strings you have seen the pattern 0^{k}1 (a run of k zeros then a one), that is more or less likely according to whether n is large or small: the pattern 0^{k}1 appears at the front of a uniform string with probability 2^{-(k+1)}, so among n distinct strings you expect to see leading-zero runs up to about log₂ n. Long leading-zero runs are a fingerprint of large cardinality.
- **Order-statistics observables.** Look at the smallest hashed values. If X = min over the set of uniform [0,1] values, then E[X] = 1/(n+1), so 1/X is a rough estimate of n. This is the basis of the MinCount / Bar-Yossef line.

Either kind can be maintained in one or a few registers. But a single observable is highly variable — one bit-pattern record or one minimum is one random draw — so it only pins n to within a binary order of magnitude. The entire game past this point is **variance reduction**: combine many such observations cheaply.

## Baselines

**Flajolet–Martin probabilistic counting with stochastic averaging (PCSA), 1985.** The first practical answer, and the direct ancestor.

The core observable: after fixing an orientation for the remaining hash bits, let r(y) be the zero-based position of the first 1-bit, equivalently the length of the initial zero run; the pattern that sets position i has probability 2^{-(i+1)}. Maintain a bit-vector BITMAP[0..L-1], initially zero, and for every element set BITMAP[r(hash(x))] := 1. After processing n distinct values, BITMAP[i] is almost surely 1 for i well below log₂ n and almost surely 0 for i well above it, with a noisy fringe around log₂ n. Let R be the position of the *leftmost zero* in the bitmap. The analysis (Mellin transform, residues) gives

  E[R] ≈ log₂(φ·n),  φ = 0.77351…  (the "magic constant"),  σ(R) ≈ 1.12 bits,

so 2^{R}/φ estimates n but a single bitmap is good only to about ±1 binary order of magnitude — far too coarse.

The fix that makes it usable is **stochastic averaging**, and it is the load-bearing idea. To average m independent observations *without* computing m hash functions per element (which would multiply CPU cost by m and require a large family of independent hashes, for which no construction is known), use the hash itself to scatter elements into m groups: pick the bucket a = h(x) mod m from part of the hash, and feed the *remaining* bits of h(x) to r to update only BITMAP_a. About n/m distinct elements land in each bucket. Compute each bucket's leftmost-zero position R_j, let S = Σ_j R_j, and return

  Z = (m/φ)·2^{S/m}.

Averaging the R_j in the exponent makes Z proportional to the geometric mean of the per-bucket 2^{R_j}, and (1/φ)·2^{S/m} estimates n/m, so m·that estimates n. Standard error improves as 0.78/√m. With m=64 bitmaps of L=32 bits each (about 256 bytes) this gives roughly 10% accuracy and counts past 10^9.

Its gap: memory is m·L bits — a *whole bitmap* per bucket — when almost all of what a bitmap holds is the single number R_j. That is wasteful; the bitmap stores L bits to recover one log₂ n-sized quantity.

**LogLog / SuperLogLog (Durand–Flajolet, 2003).** Keep the same bit-pattern family but replace the bitmap's leftmost-zero statistic by a different per-bucket statistic: the **maximum** ρ seen, where ρ = r + 1 is the one-based leftmost-1-bit position. With a finite suffix of K hash bits, the all-zero suffix is assigned ρ = K + 1. A single small register M[j] = max ρ over the bucket records the largest leading-zero run. The maximum leading-zero run in a bucket of about n/m elements is close to log₂(n/m), so the register value is on a logarithmic scale; storing that integer takes about log₂ log₂ n + O(1) bits — roughly 5 bits, a "small byte." That collapses memory from a bitmap to a doubly-logarithmic register: hence "LogLog." The estimator is

  E = α_m · m · 2^{(1/m)·Σ_j M[j]},

i.e. the arithmetic mean of the registers in the exponent, which is the **geometric mean** of the 2^{M[j]}, scaled by a bias-correction constant α_m (≈ 0.39701 in the limit). The standard error is 1.30/√m. A refinement, SuperLogLog, discards the largest ~30% of registers before averaging to cut variance toward ~1.05/√m, but the truncation makes it resist clean analysis.

Its gap: 1.30/√m is noticeably worse than the 1.00/√m the order-statistics family reaches, and the cause is identifiable — the per-bucket quantity 2^{M[j]} has a slowly decaying right tail, and the geometric mean (arithmetic mean in the exponent) still lets an occasional freak-large register pull the estimate up. The averaging rule, not the observable, is leaving accuracy on the table.

**MinCount / order-statistics estimators (Bar-Yossef et al.; Giroire).** Maintain the smallest hashed values; since E[min]=1/(n+1), the minima estimate 1/n, and averaging m of them (again by stochastic splitting) gives standard error about 1.00/√m. This is the competing family. A general lower bound (Chassaing–Gérin) places a 1/√m-scale floor on a wide class of order-statistics estimators, so ~1/√m is the accuracy scale to compare against in that line of work. Chassaing–Gérin also observe that means which down-weight the tail (harmonic-type means) tame slowly-decaying right tails and act as variance reducers.

## Evaluation settings

The natural yardstick is the relative error |estimate/n − 1| as a function of true cardinality n, measured over streams of uniform random data (ideal multisets) and over real files (e.g. distinct lines of system text, distinct keys of a relation, distinct flow identifiers of a packet trace). The metric of record is the **standard error**: the standard deviation of the estimate divided by n, reported as a percent for a given register count m, together with the **bias** (ratio of mean estimate to true n). One sweeps n across many binary orders of magnitude — from a handful up to and beyond 10^9 — at fixed memory, and reads off accuracy versus memory: error against bytes used, equivalently the constant c in c/√m for m registers. Memory is counted in registers × bits-per-register; running time is dominated by the hash, so cost is reported as a small multiple of a bare scan.

## Code framework

The primitives that already exist: a streaming loop over the data, a fixed finite hash, stochastic bucket splitting, per-bucket register storage, and register-wise merging. The open slots are the per-element bit-pattern statistic, the way a bucket accumulates that statistic, and the estimator that combines the registers.

```python
import math

HASH_BITS = 32
HASH_RANGE = 1 << HASH_BITS

def hash32(value):
    # fixed pseudo-uniform hash D -> {0,1}^32
    pass

class DistinctCounter:
    def __init__(self, p):
        self.p = p
        self.m = 1 << p                 # number of buckets
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
        # fixed stochastic split from the hash bits
        pass

    def _remaining_bits(self, x):
        # hash suffix left after bucket selection
        pass

    def _observable(self, w):
        # TODO: the function of the hashed suffix recorded per element
        pass

    def _combine_into_bucket(self, current, obs):
        # TODO: how a bucket accumulates its observable across elements
        pass

    def _combine_buckets(self, left, right):
        # TODO: how two shard summaries merge
        pass

    def estimate(self):
        # TODO: turn the m bucket summaries into a cardinality estimate
        pass

    def count(self):
        return self.estimate()
```
