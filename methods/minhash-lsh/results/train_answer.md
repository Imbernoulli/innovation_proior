The problem that forces all of this is near-duplicate detection at web scale. I have a crawl of $N \approx 10^9$ pages riddled with mirrors, syndicated copies, and pages that differ only in a timestamp or an ad, and I want to collapse them. Two pages are near-duplicate when they share almost all their text, so I turn each page into a set by $w$-shingling — every contiguous run of $w$ tokens becomes a feature — and measure resemblance by the Jaccard similarity of the shingle sets, $J(A,B) = |A\cap B|/|A\cup B|$, which is $1$ for identical sets and $0$ for disjoint ones. That reduces the task to two coupled subproblems: estimate $J(A,B)$ cheaply for a pair, and across $N$ sets find the highly similar pairs without doing all the work. Both are brutal here. Each page has thousands of shingles drawn from an astronomical universe, so exact $J$ for one pair costs $O(|A|+|B|)$ and requires holding both full sets; and finding all similar pairs by brute force is $\Theta(N^2)$ intersections, about $10^{18}$ comparisons. The obvious baselines each fail on exactly one axis or both: exact set intersection over an inverted index does not bound the candidate count and chokes on popular shingles that drop a huge fraction of the corpus into one posting list; keeping an independent uniform random sample of each set and comparing the samples is biased, because two independently drawn samples almost never hit the *same* shared elements, so the measured overlap collapses toward zero; random projections preserve $\ell_2$/cosine geometry rather than set-theoretic Jaccard and still leave the all-pairs problem untouched; and tree-based exact nearest neighbor degrades to a linear scan under the curse of dimensionality. What I need is a small fixed-size sketch per set from which $J$ is recovered to a few percent, insensitive to order and multiplicity, composable under union, plus a retrieval scheme whose per-query work is sublinear in $N$ and emits a short candidate list that is then verified exactly. That few-percent slack is precisely what buys the sublinearity.

I propose MinHash for the sketch and Locality-Sensitive Hashing by banding for the retrieval, and the two halves are the same idea seen twice. The lesson from the failed sampling baseline is that the two sketches must be drawn through a *shared* source of randomness keyed to the elements, not the sets, so that a shingle present in both sets is selected the same way in both. Hashing supplies exactly this: push every shingle through a hash function and treat the output as a uniform random value. Identical shingles hash identically, so any statistic over the bag of hashed values is really a function of the set — repeats land on themselves, order is irrelevant — and the randomness is now tied to the element. The minhash is then the order statistic that turns this into a Jaccard estimator. Fix a random permutation $\pi$ of the universe (equivalently, hash each element and look at who hashes smallest) and define $h(S)$ as the element of $S$ with the smallest hash. The element of $A\cup B$ with the global smallest hash is uniform over $A\cup B$; if it lies in $A\cap B$ it is simultaneously the minimum of $A$ and of $B$, so $h(A)=h(B)$, and if it lies in only one set the other set's minimum is some later element, so they disagree. Therefore
$$\Pr[h(A)=h(B)] = \frac{|A\cap B|}{|A\cup B|} = J(A,B).$$
The collision probability *equals* the target, not approximately. The careful row-classification check confirms it: restricting to columns $A,B$, every universe element is type $X$ (in both), $Y$ (in exactly one), or $Z$ (in neither); scanning in permuted order and ignoring the $Z$ rows that can be no one's minimum, the first non-$Z$ element is an $X$ — forcing agreement — with probability $|X|/(|X|+|Y|) = J$. The minimum is the right statistic precisely because the argmin over $A\cup B$ is uniform over $A\cup B$; a sum or median gives no agree/disagree event with probability $J$ at all.

One minhash is just a single $\mathrm{Bernoulli}(J)$ coin, worthless as a point estimate, so I take $k$ independent permutations $h_1,\dots,h_k$, form the signature $(h_1(S),\dots,h_k(S))$, and estimate
$$\hat J = \frac{1}{k}\,\#\{\,i : h_i(A)=h_i(B)\,\}.$$
This is an average of $k$ i.i.d. $\mathrm{Bernoulli}(J)$ indicators, so it is unbiased with $\mathbb{E}[\hat J]=J$ and $\mathrm{Var}(\hat J)=J(1-J)/k$, a standard error of $O(1/\sqrt{k})$ that needs $k=O(1/\varepsilon^2)$ — a couple hundred hashes for a few percent — with no bias constant to chase. A literal random permutation of billions of rows is unstorable, so I simulate each one by cheap universal hashing: hash the shingle bytes to a 32-bit value $v$, then map $v \mapsto (a_i\cdot v + b_i) \bmod p$ reduced into a 32-bit range, with $p$ the Mersenne prime $2^{61}-1$ (so the modulus is cheap) and random $a_i \in [1,p)$, $b_i \in [0,p)$. These $a\cdot x + b \bmod p$ maps are pairwise-independent stand-ins for permutations, approximately min-wise independent, and the finite-range collisions are ordinary implementation noise rather than a change of estimator. The whole sketch is then $k$ running minima: initialize each slot to $+\infty$ (the max 32-bit value), and for each element update slot $i$ to $\min(\text{slot}_i, (a_i v + b_i)\bmod p)$. This is one pass, $O(k)$ per element, and crucially the signature of $A\cup B$ is the elementwise minimum of the two signatures — so sketches merge under union, letting me shard a giant set, sketch the pieces, and combine.

That solves cheap pairwise Jaccard but not the $N^2$ pairs, since I have merely shrunk the per-comparison cost from $O(\text{set size})$ to $O(k)$. I must make similar sketches *find each other* without an all-pairs sweep, which means hashing the sketches into buckets so two sets are a candidate only if they collide somewhere. Hashing the full $k$-number signature to one bucket is too strict — collision needs all $k$ minhashes to agree, probability $J^k$, vanishing even at $J=0.9$ — while a table per single coordinate is too loose, with per-table collision probability $J$ flooding the list with dissimilar pages. The fix is to compose the two: split the length-$k$ signature into $b$ bands of $r$ rows, AND within a band (a band matches only if all $r$ coordinates agree, probability $s^r$) and OR across bands (candidate if at least one band matches). A pair fails to be a candidate only when every band fails, so with $s=J$,
$$\Pr[\text{candidate}] = 1 - (1 - s^{r})^{b}.$$
The $s^r$ is the AND-sharpening that slams moderate similarities toward zero; the $1-(1-\cdot)^b$ is the OR-recovery that gives a truly similar pair $b$ independent chances to match. The product is an S-curve — nearly flat and low for small $s$, a steep rise through a threshold, saturating near $1$ for high $s$ — exactly the separator I wanted. The exact halfway point solves $1-(1-s^r)^b = 1/2$, i.e. $s = (1-2^{-1/b})^{1/r}$, and the practical design rule drops the constant to $s^\* \approx (1/b)^{1/r}$. Tuning is then direct: more bands $b$ lowers the threshold and raises recall at the cost of more candidates to verify, while longer bands $r$ raise the threshold and tighten precision and speed at the cost of borderline misses. With $k=100$, $b=20$, $r=5$, for instance, a $40\%$-similar pair is a candidate under $19\%$ of the time while an $80\%$-similar pair is a candidate over $99.96\%$ of the time, and only about one in $3000$ truly-$80\%$ pairs slips through. I implement each band as its own hash table: hashing the $r$-tuple of every set's band and dropping its id there is one linear pass of $N\cdot b$ insertions, and a query inspects only its $b$ buckets and unions the colliding ids — those are the candidates, which then get the exact $O(k)$ verification. False positives are exactly why that verification step stays.

The same construction also makes single-query nearest neighbor provably sublinear, because the minhash is a *locality-sensitive* family: $H$ is $(r_1,r_2,p_1,p_2)$-sensitive if distance $\le r_1$ collides with probability $\ge p_1$ and distance $\ge r_2$ with probability $\le p_2$, where $p_1>p_2$, and minhash is exactly this for Jaccard distance $d=1-J$ since one minhash agrees with probability $s=1-d$. The $r$-way AND turns a probability $p$ into $p^r$ (crushing the low $p_2$ toward $0$) and the $b$-way OR turns it into $1-(1-p)^b$ (pulling the high $p_1$ toward $1$); AND-then-OR is the banding, and the S-curve is the generic signature of this amplification for any LSH family. For the query structure I build $L$ tables, each keyed by $g_j(x)=(h_1(x),\dots,h_k(x))$, the concatenation that does the AND. Choosing $k=\log N/\log(1/p_2)$ makes a far point's per-table collision probability $p_2^k \approx 1/N$, while a near point survives one table with probability $\ge p_1^k = N^{-\rho}$ where
$$\rho = \frac{\log(1/p_1)}{\log(1/p_2)}, \qquad 0 < \rho < 1.$$
Taking $L=O(N^{\rho})$ tables catches the near point with constant probability — $(1-N^{-\rho})^{N^\rho}\approx e^{-1}$ — while expected far-point collisions in the queried buckets stay $N\cdot L\cdot p_2^k = L$, so I examine $O(N^\rho)$ candidates. Query cost is $O(N^\rho)$ hash and distance evaluations and space is $O(d\cdot N + N^{1+\rho})$, sublinear; for the bit-sampling family with approximation factor $c$ one has $\rho \le 1/c$, so a $2$-approximate query costs about $\sqrt N$ instead of $N$. The curse of dimensionality is escaped by accepting an approximation and paying only $N^\rho$. The two halves thus lock together: the minhash that gives the unbiased Jaccard estimate *is* a locality-sensitive function for Jaccard distance, and AND-then-OR amplification both generates the short candidate list (in the $b$-bands form) and answers single queries in $N^\rho$ time (in the $L$-tables form).

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
