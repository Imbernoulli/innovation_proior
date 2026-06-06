# Context — Estimating set similarity and finding near-duplicates at web scale

## Research question

Given an enormous collection of objects — billions of web documents in a crawl, each turned into a *set* of features — answer two coupled questions cheaply:

1. **Pairwise similarity.** For two sets A and B, estimate their Jaccard resemblance J(A,B) = |A∩B|/|A∪B| without scanning the full sets each time. The sets are large (a document yields thousands of features), the universe of possible features is astronomical, and exact J(A,B) costs O(|A|+|B|) per pair.
2. **Near-neighbor retrieval.** Across N objects, find the pairs (or, per query, the items) whose similarity exceeds a threshold — near-duplicate pages, plagiarized passages, mirror sites — *without* the brute-force Θ(N²) comparison of all pairs, and in high dimension where space-partitioning trees collapse under the curse of dimensionality.

What a solution must achieve: a **small, fixed-size sketch** per object from which J is recovered to a few percent; a sketch that is **insensitive to multiplicity and order** (it depends only on the underlying set); sketches that **compose under union**; and a retrieval scheme whose per-query work is **sublinear in N**, generating a short candidate list that is then verified exactly. A few percent estimation error and a small, tunable false-positive/false-negative rate are acceptable — that slack is what buys the sublinearity.

Why it matters: a web search engine must purge near-duplicate pages from its index and its result lists; copyright and plagiarism systems must find documents that substantially contain one another; clustering and de-duplication of massive corpora, and approximate nearest-neighbor search over high-dimensional feature sets (audio fingerprints, genome k-mers, user-item profiles), all reduce to "estimate set similarity cheaply, then retrieve the similar ones without touching every pair."

## Background

**From documents to sets: shingling.** A document is turned into a set by *w-shingling*: take every contiguous subsequence of w tokens (or characters) as a feature. Two documents that share most of their text share most of their shingles, and their textual resemblance becomes the Jaccard similarity of their shingle sets. Larger w makes the sets sparser and the match stricter. This converts "compare documents" into "compare sets of tokens."

**Jaccard similarity and distance.** For finite sets, J(A,B) = |A∩B|/|A∪B| = |A∩B|/(|A|+|B|−|A∩B|), ranging in [0,1]: 0 for disjoint sets, 1 for identical ones. The complementary Jaccard distance d(A,B) = 1 − J(A,B) is a genuine metric on finite sets. This is the quantity to estimate and to retrieve on.

**The characteristic matrix.** Stack all sets as columns over the universe of elements (rows): M[r,c] = 1 iff element r ∈ set c. It is extremely sparse — each document touches a tiny fraction of all possible shingles. Any set-similarity computation is a computation on the columns of this conceptual matrix; we never materialize it, but it frames the analysis.

**Why exact methods don't scale on either axis.** Computing J(A,B) exactly means intersecting two large sets — O(|A|+|B|) with hashing, and you must hold both sets. Worse, finding *all* similar pairs by computing J for every pair is Θ(N²) intersections; at N = 10⁹ that is 10¹⁸ comparisons, hopeless. The two costs — per-pair set size, and number of pairs — must *both* be attacked.

**Why sampling a few elements is treacherous.** Keeping a uniform random subset of each set and comparing the subsets does not give an unbiased Jaccard estimate in general: the overlap of two independent uniform samples underestimates the overlap of the full sets, because two independently drawn samples rarely sample the *same* shared elements. The fix has to make the two sketches *coordinated* — drawn so that shared elements are sampled consistently across sets — not independently per set.

**Hashing manufactures the randomness and the set-invariance.** The recurring device: push every element through a hash function and treat its output as a uniform random value. Because identical elements hash identically, any statistic computed from the *collection* of hashed values is automatically a statistic of the *set* — duplicates land on top of each other and change nothing, and order is irrelevant. Hashing simultaneously supplies the uniform randomness the probability analysis needs and the multiplicity-/order-insensitivity the sketch requires.

**Min-wise independence.** A permutation π of the universe is *min-wise independent* (for the analysis we need) if, for every set X and every x ∈ X, the probability that x attains the minimum of π over X is exactly 1/|X| — every element of X is equally likely to be the one π ranks first. True uniform permutations have this property exactly; in practice it is approximated by hashing, and the approximation is what makes the sketches statistically reliable (Broder, Charikar, Frieze, Mitzenmacher, 1998).

**The curse of dimensionality for exact near-neighbor search.** In low dimension, k-d trees and similar space partitions answer nearest-neighbor queries in O(log N). But all such exact structures degrade to a linear scan as dimension grows — known exact data structures that beat the O(dN) scan need space exponential in d. High-dimensional exact NN is, for practical purposes, no better than brute force. The escape is to *approximate*: accept a c-factor slack in the returned distance and ask only for a point within c× the true nearest distance, which opens the door to sublinear-query randomized structures (Indyk, Motwani, 1998).

## Baselines

**Exact set intersection / inverted-index overlap.** Represent each set explicitly (sorted list, hash set, or posting list) and compute |A∩B| directly. Correct and simple; the core idea is just set intersection, and an inverted index can find sets sharing *any* element. The gap: per pair it costs O(|A|+|B|) and stores the full sets; finding all similar pairs is Θ(N²) intersections, or, via an inverted index, blows up on popular elements that put a huge fraction of the corpus in one posting list (every pair sharing a common stopword-shingle becomes a candidate). It does not give a constant-size sketch and does not bound the candidate count.

**Independent random sampling of each set.** Keep a fixed-size uniform random sample of each set and estimate J from the samples' overlap. Constant-size sketch, cheap to compare — but, as above, two independently drawn samples rarely hit the same shared elements, so the overlap statistic is biased and noisy as an estimate of |A∩B|/|A∪B|. The lesson it leaves: the sketches of two sets must be *coupled through a shared source of randomness*, so that the same elements are selected in both whenever they are present.

**Dimensionality-reduction / random projection for similarity.** Project high-dimensional vectors to a low-dimensional sketch (e.g. random projections preserving Euclidean or angular distance) and compare sketches. This attacks the per-comparison cost and inspires the general idea of a similarity-preserving summary, but the standard projections target ℓ₂/cosine geometry, not the set-theoretic Jaccard resemblance of sparse 0/1 feature sets, and on their own they still leave the Θ(N²) all-pairs problem untouched.

**Tree-based exact nearest neighbor (k-d trees, ball trees).** Recursively partition space so a query prunes most of the data. Excellent in low dimension. The gap is precisely the curse of dimensionality: in high dimension the pruning fails, every branch must be explored, and query time returns to linear — useless for the thousands-of-shingles feature spaces here.

## Evaluation settings

The natural yardstick is, first, the **estimation accuracy** of the similarity sketch: the standard error of the Jaccard estimate as a function of the sketch length, measured over pairs of sets with known true J — reported as error versus number of hash functions (memory), with the estimator's bias. Second, the **retrieval quality and cost** of the near-neighbor scheme: for a chosen similarity threshold, the false-negative rate (truly similar pairs missed) and false-positive rate (dissimilar pairs surfaced as candidates), as functions of the banding parameters; the **candidate-set size** and per-query work as a function of N (the quantity that must be sublinear); and the total time to find all near-duplicate pairs in a corpus versus the brute-force all-pairs baseline. Standard data: large document crawls turned into shingle sets, and synthetic set collections with controlled overlap so the true J is known. Memory is counted as hash functions × bytes per value plus the hash tables; time is dominated by hashing each element once.

## Code framework

The primitives that already exist: a fixed pseudo-uniform hash from elements to a 32-bit range; a family of cheap pairwise-independent maps a·x+b mod p to stand in for "random permutations" of the universe; a one-pass reduction over a set's elements; and a dictionary keyed by a tuple of values, for bucketing. The open slots are the per-element statistic the sketch records, how the sketch estimates similarity, and how sketches are turned into a sublinear candidate-generation index.

```python
import numpy as np

MERSENNE_PRIME = (1 << 61) - 1          # p, for the a*x+b mod p maps
MAX_HASH = (1 << 32) - 1                # reduce into a 32-bit range
HASH_RANGE = 1 << 32

def hash32(value):
    # fixed pseudo-uniform hash: element -> {0,...,2^32-1}
    pass

class SetSketch:
    def __init__(self, num_perm=128, seed=1):
        self.num_perm = num_perm
        # k cheap pairwise-independent maps standing in for random permutations
        gen = np.random.RandomState(seed)
        self.a = gen.randint(1, MERSENNE_PRIME, size=num_perm, dtype=np.uint64)
        self.b = gen.randint(0, MERSENNE_PRIME, size=num_perm, dtype=np.uint64)
        self.sketch = self._init_sketch(num_perm)

    def _init_sketch(self, num_perm):
        # TODO: the initial per-slot value the one-pass reduction starts from
        pass

    def update(self, value):
        hv = np.uint64(hash32(value))
        permuted = (self.a * hv + self.b) % MERSENNE_PRIME & np.uint64(MAX_HASH)
        # TODO: the per-element statistic each slot records across the set
        self.sketch = self._combine(self.sketch, permuted)

    def _combine(self, current, permuted):
        # TODO: how a slot accumulates its statistic over the set's elements
        pass

    def merge(self, other):
        # TODO: combine two sketches into the sketch of the union
        pass

    def similarity(self, other):
        # TODO: estimate J(A,B) from the two sketches
        pass


class SimilarityIndex:
    """Generate a short candidate list of near-neighbors in sublinear work."""
    def __init__(self, threshold, num_perm=128):
        self.num_perm = num_perm
        # TODO: choose how to slice / amplify the sketch so that the
        #       collision probability is a sharp threshold function of J
        self.b, self.r = self._choose_amplification(threshold, num_perm)
        self.tables = [dict() for _ in range(self.b)]

    def _choose_amplification(self, threshold, num_perm):
        # TODO: pick the amplification parameters placing the threshold at `threshold`
        pass

    def _keys(self, sketch):
        # TODO: derive the per-slice bucket keys from a sketch
        pass

    def insert(self, key, sketch):
        for t, h in zip(self.tables, self._keys(sketch)):
            t.setdefault(h, []).append(key)

    def query(self, sketch):
        # TODO: collect candidates colliding in any slice; caller verifies exactly
        pass
```
