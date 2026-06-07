# Synthesis — Reservoir Sampling

## Problem
Draw a uniform sample of k items from a stream of unknown, possibly huge length N, in
ONE pass and O(k) memory. N is not known in advance and may be too large to store or
even to count in a cheap first pass (tape/disk, data stream).

## Why the obvious approaches fail (the walls)
- Know N -> pick k random indices in {1..N}: needs N known, or a counting pass. Banned.
- Store everything, sample at the end: O(N) memory. Banned.
- Keep each item with fixed prob p: sample size is Binomial(N,p), not exactly k, and
  depends on N which we don't know. Banned.

## Algorithm R (Waterman; Vitter 1985)
- Fill: first k items go straight into reservoir R[1..k].
- For the i-th item (i>k): pick j uniform in {1..i}. If j<=k, R[j]:=x_i; else discard.
  Equivalent: accept x_i with probability k/i, replacing a uniformly random slot.
- INDUCTION (each item in reservoir w.p. k/n at end):
  - After i items, claim each of the i seen is present w.p. k/i. Base i=k: all k present, k/k=1.
  - Step i->i+1: new item accepted w.p. k/(i+1) (correct target). An old item x (present w.p. k/i)
    survives iff (new item rejected) OR (accepted but didn't evict x):
    P(survive) = (k/i) * [ (1 - k/(i+1)) + (k/(i+1))*(1 - 1/k) ]
               = (k/i) * [ 1 - k/(i+1) + k/(i+1) - 1/(i+1) ]
               = (k/i) * (1 - 1/(i+1)) = (k/i) * (i/(i+1)) = k/(i+1).  QED
- Cost: O(N) time, O(N) RNG draws (one per item). That is the inefficiency Vitter attacks.

## The framework realization (Vitter 1985, Sec 3)
- ANY one-pass algorithm must be a reservoir algorithm: after t items, item t+1 must be
  accepted with prob exactly k/(t+1) (else not a valid sample). So the only freedom is HOW
  you generate the acceptances. The wasted work in R is the per-item coin flip.
- Define S = S(k,t) = number of items skipped before the next acceptance. Generate S, skip
  S items, accept the next, repeat. Skip distribution (exact, Vitter eq 3.1):
    F(s) = P(S <= s) = 1 - (t+1-k)^{(s+1) falling} / (t+1)^{(s+1) falling}
                     = 1 - [ (t-k+1)(t-k+2)...(t-k+1) ... ] ... ; equivalently the product form
    F(s) = 1 - prod_{j=1..s+1} (t+1-k+? ) -- in plain terms 1 - product of (1 - k/(t+1+m)) over the skipped items.
  Number of acceptances total ~ k(1+ln(N/k)) = k(H_N - H_k)+... so optimal time is O(k(1+log(N/k))).
- Algorithm X: invert F by sequential search, O(S) per skip but only ONE RNG call per skip.
- Algorithm Z: generate S in O(1) by von Neumann rejection -> overall optimal O(k(1+log(N/k))).

## Algorithm L (Li 1994) — the clean key/threshold view (used for final code)
- Key view (also Wikipedia/Florian "random tags"): give each item an iid key u_i ~ U(0,1).
  The k items with the SMALLEST keys form a uniform sample of size k. (Equivalently keep k
  largest of (1-u); the design choice of smallest vs largest is cosmetic.)
- Online: keep the k smallest keys in a heap; the current threshold is the LARGEST of those k
  smallest keys, call it w. A new item is accepted iff its key < w, probability w.
- Distribution of w: w is the max of the k retained keys. Max of k iid U(0,1) has CDF x^k, so
  w is distributed as U^{1/k}  (inverse-CDF: x^k = u  =>  x = u^{1/k}). After an acceptance the
  retained set re-tightens; the new threshold = old w * U^{1/k}  (product of such factors).
- Since each subsequent item is independently accepted w.p. w, the GAP (number skipped) before
  the next acceptance is geometric with success prob w:  P(skip = g) = (1-w)^g * w.
  Inverse-CDF of geometric: skip = floor( log(U) / log(1-w) ).
- Wikipedia pseudocode (verbatim):
    W := exp(log(random())/k)
    while i <= n:
       i := i + floor(log(random())/log(1-W)) + 1
       if i<=n: R[randomInteger(1,k)] := S[i]; W := W * exp(log(random())/k)
  (exp(log(U)/k) == U^{1/k}.)
- Cost: O(k(1+log(n/k))) RNG calls — optimal, matches Algorithm Z, far simpler.

## Weighted (A-Res / A-ExpJ; Efraimidis & Spirakis 2006) — without replacement, weight-proportional
- Generalize the key trick to weights: key k_i = u_i^{1/w_i}, u_i~U(0,1). Keep the k LARGEST keys.
  Theorem: this is a weighted random sample without replacement (round-r selection prob = w_i / sum of remaining weights).
- A-Res: size-k min-heap; threshold T = min key; accept item if u^{1/w} > T, replace min. O(n) RNG.
- A-ExpJ: jump version. Let S_w = sum of skipped weights until next entry; S_w ~ Exponential.
  X_w = log(r)/log(T_w) is the weight budget; skip items until cumulative weight reaches X_w.
  On accept: t_w = T_w^{w_i}, r2 = random(t_w,1), new key = r2^{1/w_i}; update T_w. RNG: O(k log(n/k)).

## Design-decision -> why
- Fill first k, not random first k: must seed the reservoir; the first k ARE a valid sample of the first k.
- Accept prob k/i (R): exactly the prob the i-th item belongs in a size-k sample of i items -> keeps invariant.
- Evict UNIFORM slot: keeps the surviving items exchangeable/uniform; non-uniform eviction breaks the induction.
- Skip variable S (framework): the per-item coin flips of R are pure waste once you realize acceptances are rare
  (only ~k log(N/k) of them); generate the gap directly.
- Geometric gap (L): acceptances are iid-Bernoulli(w) across a run with fixed threshold w -> gap is geometric.
- w = U^{1/k} and multiplicative update: w IS the k-th order statistic threshold; tracking the single scalar w
  (not the whole key array) is what gives O(k) memory and O(1) work per accept.
- Key = u^{1/w} (weighted): P(key < x) = x^{w}; the largest key among a set is then weight-proportional ->
  selecting top-k keys = weighted sampling without replacement.

## Canonical code
methods/reservoir-sampling/code/reservoir.py — ReservoirR, ReservoirL, a_res, a_expj.
Verified: R and L give uniform counts over N=10,k=3 (~18000 each, expected k*trials/N).
A-Res/A-ExpJ k=1 give weight-proportional fractions {0.067,0.133,0.2,0.267,0.333}.

## Self-account (HARD GATE item 2): NONE FOUND
No first-person Vitter retrospective / award lecture / interview located this run (web search
returned only the primary paper + third-party explainers). The trace is reconstructed from the
primary source (whose §2-3 themselves narrate the motivating reasoning) + the Knuth/Waterman
antecedent. See refs/antecedents_attribution.md.

## Sources
- Vitter 1985 PDF (refs/vitter1985.txt): Algorithm R, framework, F(s), Algorithm X/Y/Z.
- Efraimidis & Spirakis encyclopedia entry (refs/efraimidis_enc.txt): A, A-Res, A-ExpJ verbatim.
- Wikipedia Reservoir sampling: Algorithm R + L + A-Res pseudocode, induction proof, key/threshold view.
- Li 1994 (TOMS 20(4)) Algorithm L optimal O(n(1+log(N/n))).
- Richardstartin blog: Java Algorithm R/L; Florian.github.io: random-tags derivation.
