# Context: estimating frequency moments of a stream in small space

## Research question

A sequence of elements arrives one at a time: `A = (a_1, a_2, ..., a_m)`, each `a_j` drawn from a universe `N = {1, ..., n}`. Let `m_i` be the number of times value `i` occurs. The *frequency moments* are

```
F_k = sum_{i=1..n} m_i^k.
```

`F_0` is the number of distinct values, `F_1 = m` is the stream length, and `F_2 = sum_i m_i^2` is the *repeat rate* — equivalently the size of the self-join `r ⋈ r` of a relation `r` whose join-attribute value `i` has frequency `m_i`, and the quantity behind Gini's index of homogeneity and the statistical surprise index. The goal: given the stream in **one pass**, output a number `Y` that is within a relative error `λ` of `F_k` with failure probability at most `δ`, while using as little working memory as possible — ideally polylogarithmic in `n` and `m`.

This matters because frequency moments summarize the *skew* of a data set, which drives decisions in parallel databases (data partitioning, join-algorithm selection) and query optimization. The defining constraint is that the elements stream by and cannot be stored: any solution must commit to a tiny summary and update it incrementally as records are inserted. Storing the data in external memory to compute moments exactly imposes a prohibitive access- and update-time overhead, so the moments must be estimated inside a small, fast register set.

## Background

The exact computation of `F_k` is easy if space is free: maintain a full histogram, a counter `m_i` for every value `i`, which costs `Θ(n log m)` bits. That is exactly the cost we want to avoid. The interesting regime is *sublinear* space — far less than one counter per universe element — which forces randomization and approximation.

Several pieces of the toolkit already exist:

- **Approximate counting in `O(log log m)` bits (Morris 1978).** To count up to `m` events one would normally need `log m` bits. Morris's idea: store not the count but (roughly) its logarithm. Keep a small register `v`; on each event, increment `v` only with probability about `2^{-v}` (so the register advances ever more slowly), and estimate the count as about `2^v - 1`. The expected value of the represented count is maintained exactly, and the relative error is controlled. This shows a single counter can be squeezed from `log m` to `log log m` bits — the right tool whenever a count `r` or a running sum must be held to limited accuracy under extreme memory pressure.

- **Hashing-based distinct counting (Flajolet & Martin 1985).** To estimate `F_0` they hash each incoming value and watch the *bit patterns* of the hashed values. For a uniformly distributed hash output, a trailing run of `r` zero bits (a pattern `0^r 1`) appears with probability `2^{-r}`; among `F_0` distinct values the largest such run `R` observed is about `log_2 F_0`, so `2^R` estimates `F_0`. A `BITMAP` of `O(log n)` bits records which runs have occurred. The estimator depends only on the *set* of distinct values, not on their multiplicities — duplicates are automatically idempotent — so `F_0` is estimated in `O(log n)` bits. The published analysis, however, leans on the assumption that an explicit hash family with very strong (essentially ideal) uniformity and independence is available, which is not something one can actually exhibit cheaply.

- **Bounded independence and small sample spaces.** A function or sign assignment is `k`-wise independent if every `k` of its outputs behave as if fully independent. Full independence over `n` points costs `Ω(n)` random bits to store; but `k`-wise independent families for small fixed `k` can be generated from a seed of only `O(log n)` bits. Concretely, a sequence of `±1` values that is four-wise independent — every four coordinates take each of the `16` sign patterns on a `1/16` fraction of the sample space — can be built from the parity-check matrix of a BCH code (an orthogonal array of strength `4`), needing only an irreducible polynomial of degree `d` over `GF(2)` with `2^d` just above `n`; each coordinate is then computable in `O(log n)` space. Degree-`(k-1)` polynomials over finite fields give `k`-wise independent field values by the same interpolation principle.

- **Concentration tools.** Chebyshev's inequality turns a variance bound into a deviation bound: `P(|X - E X| > λ E X) ≤ Var(X) / (λ E X)^2`. The Chernoff bound says the median of many independent trials, each correct with probability `≥ 3/4`, is wrong only with probability exponentially small in the number of trials. Together they support a two-stage amplification: average to shrink variance, then take a median to make the failure probability tiny.

- **Communication complexity as a lower-bound engine (Yao; Babai–Frankl–Simon; Kalyanasundaram–Schnitger; Razborov).** The `ε`-error probabilistic communication complexity `C_ε(f)` of a Boolean function, and its distributional analogue `D_ε(f | μ)`, lower-bound how many bits two (or many) parties must exchange. The **Disjointness** function — do two subsets of `{1,...,n}` intersect? — was shown to require `Ω(n)` communication. A small-space streaming algorithm yields a cheap communication protocol (split the stream between parties, pass the memory contents across the cut), so communication lower bounds translate into space lower bounds.

## Baselines

The prior art a new estimator would be measured against:

- **Exact histogram.** Maintain every `m_i`. Cost `Θ(n log m)` bits, one pass. Correct but linear in `n` — the thing to beat. Even approximating each `m_i` à la Morris only reduces this to `O(n log log m)` bits: still linear in `n`.

- **Morris approximate counter (F_1).** `O(log log m)` bits to approximate the stream length `F_1 = m` within a constant factor. Tight for `F_1`, but says nothing about higher moments where collisions between distinct values matter.

- **Flajolet–Martin (F_0).** `O(log n)` bits to approximate the number of distinct elements. The gap it leaves: its analysis presumes an idealized hash family that is not cheaply constructible, and it addresses only `F_0`, not the squared-frequency structure of `F_2` or higher `F_k`.

- **Sampling-based skew estimators (Haas et al.; Whang et al.).** Estimate `F_0` and skew by sampling tuples, sometimes choosing the estimator adaptively based on a skew measure. Useful in databases but sample-based rather than a single small streaming sketch, and not a one-pass small-space guarantee for `F_2` itself.

The open gap across all of these: a *single-pass, polylogarithmic-space, provable `(λ, δ)` approximation of `F_2`* — and more generally of `F_k` — with hash families one can actually build, plus matching evidence of where small space becomes impossible.

## Evaluation settings

The natural yardstick is the streaming model itself: a single left-to-right pass over a sequence of up to `m` tokens from `[n]`, with the working memory measured in bits as a function of `n`, `m`, the relative error `λ`, and the failure probability `δ`. The quantities to estimate are `F_0`, `F_1`, `F_2`, general `F_k`, and `F_∞* = max_i m_i`. Representative data are database relations whose join-attribute values stream by (so that `F_2` is a self-join size), and skewed demographic-style multisets. The protocol allows randomization (a random seed chosen before the pass) and reports an estimate at the end; a small constant number of passes is also of interest. The figures of merit are worst-case space and the `(λ, δ)` guarantee.

## Code framework

A minimal streaming harness has a one-pass iterator, finite-field arithmetic for seeded bounded-independence hashing, and a median-of-means combiner around a compact summary object.

```python
import math, random, statistics

FIELD_BITS = 64
FIELD_MASK = (1 << FIELD_BITS) - 1
REDUCTION = 0x1B  # x^64 + x^4 + x^3 + x + 1


def gf_mul(a, b):
    """Multiply in a fixed binary field."""
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
    """A seedable hash from a small (bounded-independence) family.
    Seed is O(log n) bits; each output is computable in O(log n) space."""
    def __init__(self, rng):
        pass  # TODO: pick the family and the seed

    def __call__(self, x):
        pass  # TODO: map a universe element x to the coefficient the sketch needs


class StreamSummary:
    """One-pass summary of a stream over [n] with tiny state, an O(1)
    incremental update, and a final read-out."""
    def __init__(self, eps, delta, seed=0):
        self.rng = random.Random(seed)
        # median-of-means dimensions: s1 averages cut variance, s2 medians
        # cut failure probability. The constants depend on the estimator.
        self.s1 = None  # TODO
        self.s2 = None  # TODO
        # TODO: allocate the per-copy state and its hashes

    def update(self, token, count=1):
        pass  # TODO: the O(1) incremental update per token

    def estimate(self):
        pass  # TODO: combine the state into an (eps, delta)-estimate of F_2


def median_of_means(per_copy_values, s1, s2):
    """Average s1 values into each of s2 groups, return the median of the
    group means."""
    rows = [per_copy_values[r * s1:(r + 1) * s1] for r in range(s2)]
    return statistics.median(sum(row) / s1 for row in rows)
```
