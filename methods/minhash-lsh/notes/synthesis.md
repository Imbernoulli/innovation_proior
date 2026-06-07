# Synthesis — MinHash + LSH

## Pain point / research question
- Web-scale near-duplicate detection (AltaVista): billions of documents, want to cluster near-identical pages, detect plagiarism, dedup crawl. Each doc -> a set (of shingles). Compare by Jaccard resemblance r(A,B)=|A∩B|/|A∪B|.
- Two problems: (1) the sets are huge (each doc = thousands of shingles, universe of all shingles is astronomical) — exact Jaccard per pair is O(|A|+|B|); (2) there are N docs => N^2/2 pairs — quadratic, infeasible at N=10^9.
- Want: (a) cheap *estimate* of Jaccard from a tiny fixed-size sketch; (b) find the near-duplicate / nearest-neighbor pairs in *sub-quadratic* (sublinear-per-query) time, without comparing all pairs.

## Background / load-bearing facts (web-grounded)
- Jaccard J(A,B)=|A∩B|/|A∪B| = |A∩B|/(|A|+|B|-|A∩B|), range [0,1], =0 disjoint, =1 identical. Jaccard distance d=1-J is a metric. (Wikipedia Jaccard_index.)
- Shingling: doc -> set of w-shingles (contiguous length-w token/char subsequences). Resemblance of docs = Jaccard of shingle sets. (Broder 1997.)
- Characteristic matrix M: rows=universe elements, cols=sets, M[r,c]=1 iff element r in set c. Sparse. (MMDS ch3.)

## MinHash core (grounded: MMDS ch3, Broder 1997, Wikipedia MinHash)
- Pick random permutation π of universe. h(S)=index of first row (in π order) that S has a 1 in = argmin over S of π. = "minhash".
- THEOREM: Pr[h(A)=h(B)] = J(A,B). Proof (MMDS three-row-type argument): among the two columns, rows are type X (1,1), type Y (one 1), type Z (0,0). x=|X|=|A∩B|, x+y=|A∪B|. Scanning rows top-down in random permutation order, ignore Z. First non-Z row is X (=> minhashes agree) with prob x/(x+y); if first non-Z is Y, the set with 0 there gets its min further down => disagree. So Pr[agree]=x/(x+y)=J. Equivalent argument (Broder/Wikipedia): the min of the union is equally likely to be any element of A∪B; minhashes agree iff that min element lies in A∩B, prob |A∩B|/|A∪B|.
- Estimator: k independent permutations h_1..h_k. signature = (h_1(S),...,h_k(S)). Estimate Ĵ = (#i: h_i(A)=h_i(B))/k. Each indicator is Bernoulli(J), independent => Ĵ unbiased, Var = J(1-J)/k, std error O(1/sqrt(k)), need k=O(1/ε^2) for error ε. (Wikipedia MinHash.)
- Min-wise independence: for the proof we need that for any set X and any x in X, Pr[π puts x first in X] = 1/|X|, uniformly. True permutations are min-wise independent; in practice approximate with hash functions. (Broder-Charikar-Frieze-Mitzenmacher 1998.)
- Implementation of "random permutation": cannot literally permute billions of rows. Simulate with a random universal-hash on the row id: h_i(r) = (a_i*x + b_i) mod p, reduced mod 2^32 (a in [1,p), b in [0,p), p Mersenne prime 2^61-1). For each i keep min over the set's elements. Collisions in the hash are harmless if range >> #rows. (MMDS ch3; datasketch minhash.py: _mersenne_prime=2^61-1, _max_hash=2^32-1, phv = bitwise_and((a*hv+b)%prime, max_hash); hashvalues = minimum(phv, hashvalues).)
- Streaming/one-pass: signature computed in one pass over each set's elements (init to +inf/_max_hash, take min). Mergeable: union of two sets' minhashes = elementwise min => sketches compose under union (datasketch merge()).
- Single-permutation variant: one hash, keep k smallest values of A∪B as a random sample; estimate J from sample. (Wikipedia, "bottom-k".)

## LSH banding / S-curve (grounded: MMDS ch3.4)
- Even with a sketch, comparing all N^2/2 pairs is quadratic. Want to only *generate candidate pairs* that are likely similar.
- Banding: signature length n split into b bands of r rows (n=br). Hash each band (the r-tuple) to a bucket; one hash table per band. Two sets are a *candidate pair* if they land in the same bucket in >=1 band.
- A single row agrees with prob s (=J). Band (r rows) all-agree: s^r. Band disagrees: 1-s^r. All b bands disagree: (1-s^r)^b. >=1 band agrees (=> candidate): **P(candidate) = 1 - (1 - s^r)^b**. (MMDS 3.4.2.)
- This is an S-curve in s: low for small s, steep rise near a threshold, saturates near 1. Threshold (P=1/2) ≈ **(1/b)^(1/r)**. e.g. b=16,r=4 => threshold ≈ 1/2 (4th root of 1/16). e.g. b=20,r=5: P(s=.5)=.470, P(s=.6)=.802, slope>3 in middle. (MMDS 3.4.2, Fig 3.8.)
- Tuning: choose threshold t; pick b,r with br=n and (1/b)^(1/r)≈t. Lower threshold (bigger b / smaller r) => fewer false negatives, more candidates. Higher threshold => fewer false positives, faster. (MMDS 3.4.3.)
- AND/OR view (MMDS 3.6): (d1,d2,p1,p2)-sensitive family. r-AND: p->p^r (within band). b-OR: p->1-(1-p)^b (across bands). banding = AND then OR = 1-(1-p^r)^b. AND pushes small probs to 0, OR pushes large probs to 1; cascade to steepen S-curve arbitrarily. minhash family is (d1,d2,1-d1,1-d2)-sensitive in Jaccard distance.

## LSH for ANN / sublinear query (grounded: Indyk-Motwani 1998 via Princeton lec10)
- (r1,r2,p1,p2)-sensitive family H: d(x,y)<=r1 => Pr[h(x)=h(y)]>=p1; d(x,y)>=r2 => <=p2.
- Build L hash tables; each uses g_j(x)=(h_1(x),...,h_k(x)) (concatenation of k=AND). Store all points; query q examines buckets g_1(q)..g_L(q), brute-force-checks candidates.
- Set k = log n / log(1/p2), L = 2 n^ρ where **ρ = log(1/p1)/log(1/p2)** (in (0,1)).
- Query: O(n^ρ) hash evals + distance computes; Space: O(dn + n^{1+ρ}). For Hamming with this family ρ <= 1/c (c=approx factor) => query n^{1/c}, e.g. c=2 => sqrt(n). Sublinear. (Indyk-Motwani 1998.)
- Proof sketch: P(a true near point collides in table j) >= p1^k = n^{-ρ}; over L=2n^ρ tables, P(miss all) <= (1-n^{-ρ})^{2n^ρ} <= e^{-2} => found w.p. >=3/4. Expected #far collisions <= n*L*p2^k = L (since p2^k=1/n) => Markov, <=4L w.p. >=3/4. So examine O(L)=O(n^ρ) candidates.

## Canonical code (datasketch ekzhu/datasketch)
- minhash.py: num_perm=128 default, seed=1. permutations = (a,b) arrays, a in [1,2^61-1), b in [0,2^61-1). update(b): hv=sha1_hash32(b); phv=((a*hv+b)%(2^61-1)) & (2^32-1); hashvalues=min(phv,hashvalues). jaccard(other)=count(hashvalues==other.hashvalues)/k. merge=elementwise min. hashvalues init = ones*_max_hash.
- hashfunc sha1_hash32: struct.unpack("<I", sha1(data).digest()[:4])[0].
- lsh.py: MinHashLSH(threshold, num_perm, weights=(fp,fn)). _optimal_param: brute over b in 1..num_perm, r in 1..num_perm/b, minimize fp_weight*FP + fn_weight*FN where FP=∫_0^t (1-(1-s^r)^b) ds, FN=∫_t^1 (1-(1-(1-s^r)^b)) ds. hashranges=[(i*r,(i+1)*r)]. b hashtables. insert: Hs = byteswap(hashvalues[start:end]) per band; hashtable.insert(H,key). query: collect candidates over bands where band-key collides; union; recommend re-filtering with MinHash.jaccard.

## Design decisions -> why
- Why hash to make a *set* observable: identical elements hash identically => any function of the multiset of hashes is a function of the *set*; min over the set is permutation-invariant. Makes sketch insensitive to multiplicity/order.
- Why MIN (not, say, sum or a specific element): need an observable whose collision probability equals J. The min of a random permutation is uniform over the set; two sets' minima coincide iff the global-min element is shared => prob = |∩|/|∪| exactly. Sum/mean would not give a clean collision=Jaccard identity and wouldn't be a single comparable token.
- Why k independent hashes: one minhash is a single Bernoulli(J) — only tells agree/disagree, useless as point estimate. Average k => Var J(1-J)/k. Variance reduction by averaging, exactly like a Monte-Carlo mean.
- Why simulate permutations with universal hashing a*x+b mod p: true permutations of 10^9 rows are unstorable/unsortable; (a*x+b) mod prime is a cheap pairwise-independent map, approximately min-wise; collisions harmless when range>>n.
- Why banding not pairwise sketch compare: sketch compare is still O(N^2) pairs. Banding turns "find similar pairs" into a hashing/bucketing problem => candidates generated in ~linear time; only candidate pairs get the O(k) verify.
- Why AND-then-OR (bands): a single minhash collision prob = s is a shallow line, no threshold. AND (r rows) sharpens (s^r kills moderate s), OR (b bands) recovers recall for high s (1-(1-·)^b). Composition makes the S-curve, separating "above threshold" from "below" sharply. Tune b,r to place/steepen the threshold.
- Why ρ=log(1/p1)/log(1/p2) and L=n^ρ: choose k so far-point collisions per table ~1/n (k=log n/log(1/p2)); then near-point survival per table = p1^k = n^{-ρ}; need L≈n^ρ tables to catch it w.h.p. => query examines O(n^ρ) candidates, sublinear since p1>p2 => ρ<1.

## Uncertainty flags
- "single-permutation / bottom-k" estimator details are a refinement; I keep MinHash core as k-permutation (the canonical datasketch path) and mention bottom-k briefly.
- α/exact small-sample bias of MinHash estimator: estimator y/k is exactly unbiased (Bernoulli), no bias constant needed — simpler than HLL.
- ρ<=1/c bound is specific to the Hamming/bit-sampling family; for Jaccard the analogous family is minhash, ρ depends on (p1,p2)=(s1,s2). Stated generally.

## Self-account hunt (this run)
- No first-person "discovery anecdote" interview/oral-history exists for either Broder (MinHash) or Indyk (LSH) that I could retrieve. WHAT EXISTS instead — and is richer in omitted *reasoning*:
  - Broder, Glassman, Manasse, Zweig, "Syntactic Clustering of the Web" (DEC SRC Tech Note 1997-015): Broder's own application-side narration of the AltaVista near-duplicate problem (30M docs, 10^15 pairs, divide-compute-merge, ~800B sketch, common-shingle and super-shingle refinements). Saved to refs/. Used to ground the context's pain-point / web-scale framing.
  - Indyk, ICM 2018 survey "Approximate Nearest Neighbor Search in High Dimensions": Indyk's retrospective re-derivation of the LSH exponent line (ρ=log(P1)/log(P2), L=n^ρ/P1, bit-sampling ρ=1/c). Saved to refs/. Used to confirm the ANN derivation in reasoning.md.
- Added both to SELF_ACCOUNT_SOURCES.md.
- reasoning.md is reconstructed in-frame from the primary (Broder 1997 resemblance/containment + Indyk-Motwani 1998 via Princeton lec10 + MMDS ch3) plus these retrospectives; no anecdote was stretched.
