# Synthesis — Count-Min Sketch

## Pain point / research question
Stream of updates (i_t, c_t) building an implicit vector a of dimension n (n huge, e.g. 2^32 IP addresses).
Want point queries a_i (and range, inner product, heavy hitters, quantiles) in space sublinear in n,
update fast, with (eps, delta) accuracy: error within eps factor w.p. 1-delta.

## Tools on the table (lineage) and their gaps
1. **AMS sketch (Alon–Matias–Szegedy 1996)** — tug-of-war: random ±1 sign hash s(i), maintain z = sum_i a_i s(i),
   estimate F_2 = ||a||_2^2 by z^2. Needs **4-wise independent** hashes. Average many for variance.
   Space ~ 1/eps^2. Gap: 1/eps^2 (expensive for eps=0.01); needs strong independence (hard in hardware);
   built for norms, not per-item point queries.
2. **Count Sketch (Charikar–Chen–Farach-Colton 2002)** — d×w table, each item hashed to one cell per row AND a
   ±1 sign g(i); add g(i) per update; estimate a_i = **median_j g_j(i)·C[j,h_j(i)]**. Signs symmetrize noise (unbiased),
   so median works. Guarantee relative to **L2**: error eps·||a||_2, space **1/eps^2**. Needs sign hashes + median.
   Gap: still 1/eps^2; the L2 guarantee is the source of the 1/eps^2 (variance ~ ||a||_2^2 / w).
3. **Bloom-filter / multistage-filter intuition** — multiple hash tables, AND/min across them, only limited independence.

## Key insight that gets to 1/eps
Drop the signs. For non-negative streams, every collision can only ADD to a counter ⇒ each row's cell C[j,h_j(i)]
is an **overestimate** of a_i. So a_i ≤ C[j,h_j(i)] always; the **minimum over rows is the tightest overestimate**
and is the natural estimator (no median needed). Because error is one-sided we can use **Markov** (first-moment) instead
of Chebyshev (second-moment): expected error in one row = (sum of other items' mass)/w = (||a||_1 - a_i)/w ≤ ||a||_1/w.
Markov: P(error > e·E[error]) < 1/e. With w = e/eps, E[error] ≤ (eps/e)||a||_1, so P(one row error > eps||a||_1) < 1/e.
d independent rows (pairwise-indep hashes), all bad: (1/e)^d = e^{-d} ≤ delta ⇒ d = ln(1/delta).
First moment only ⇒ **pairwise independence suffices** (don't need 4-wise like AMS). And 1/eps not 1/eps^2.

## Exact derivation (point query, Theorem 1, non-negative case)
- Indicator I_{i,j,k} = 1 if (i≠k) ∧ (h_j(i)=h_j(k)). Pairwise indep: E[I_{i,j,k}] = P[h_j(i)=h_j(k)] ≤ 1/range(h_j) = 1/w.
  With w = e/eps this is eps/e.
- X_{i,j} = sum_{k} I_{i,j,k} a_k ≥ 0 (non-neg case). C[j,h_j(i)] = a_i + X_{i,j} so min_j C ≥ a_i (one-sided).
- E[X_{i,j}] = sum_k a_k E[I_{i,j,k}] ≤ (eps/e)||a||_1 (linearity).
- P[â_i > a_i + eps||a||_1] = P[∀j: X_{i,j} > eps||a||_1] = P[∀j: X_{i,j} > e·E[X_{i,j}]] < e^{-d} ≤ delta by Markov + indep.
- Space minimization: more generally w = e/b, d = log_b(1/delta), cost (2 + e/eps?) — actually solve d(wd)/db = 0
  with wd = (e/b)·(ln(1/delta)/ln b); minimized at b = e, giving (2 + e/eps)? The paper: cost (2 + e/ε) ln(1/δ) words... 
  re-check: setting b=e minimizes wd = (e/ε)·ln(1/δ). [paper states b=e minimizes, cost shown as (2+e/ε)ln(1/δ) — the +2 is hash storage]

## Inner product / join (Theorem 2)
(a⊙b)_j = sum_k count_a[j,k]·count_b[j,k] = a·b + sum_{p≠q, h_j(p)=h_j(q)} a_p b_q. Estimate = min_j (a⊙b)_j.
E[extra] = sum_{p≠q} P[h_j(p)=h_j(q)] a_p b_q ≤ (eps/e)||a||_1||b||_1. Markov ⇒ error ≤ eps||a||_1||b||_1 w.p. 1-delta.
Self-join (a=b) estimates F_2. Compare AMS: eps||a||_2||b||_2 but 1/eps^2 space.

## Linearity / turnstile / merge
Sketch is a linear projection: CM[j,k] = a · r_{j,k}, r_{j,k}[i]=1 iff h_j(i)=k. So:
sketch(a+b) = sketch(a)+sketch(b) (same hashes), sketch(λa)=λ·sketch(a). ⇒ deletions (negative c_t) handled,
distributed merge by adding sketches. General (signed a) case: use **median** (CM_PointMed) since X can be negative;
needs larger d.

## Range / heavy hitters / quantiles (hierarchical CMH)
- Range query sum_{l..r}: keep log n (or U/gran levels) sketches, level k sketches the dyadic-aggregated vector
  a^k[j] = sum_{i=j2^k}^{(j+1)2^k -1} a_i. Any [l,r] = ≤ 2 log n dyadic ranges ⇒ answer with O(log n) point queries.
  Total space O((log^2 n)/eps · log(1/delta)). Error eps||a||_1, right-bound certain, left w.p. 1-delta.
- Heavy hitters (a_i ≥ phi||a||_1): recursively descend the dyadic tree; at each level query the node's count;
  if ≥ thresh recurse into its children; at leaves output the heavy items. (CMH_recursive / CMH_FindHH.)
- Quantiles: binary-search range sums for the point where prefix mass = phi||a||_1 (CMH_FindRange / CMH_Quantile).
- High levels of the hierarchy kept EXACT (freelim) since a dyadic level with few nodes is cheaper exact than sketched.

## Conservative / minimal update (Estan–Varghese 2002)
On positive-only updates: instead of incrementing all d cells by c, set each to max(C[j,h_j(i)], â_i + c)
where â_i is current estimate. Never increases the estimate beyond what's needed ⇒ point estimate still ≥ a_i,
error never worse, often an order of magnitude better. BUT no deletions, and per-update cost ~doubles.

## Hash function (canonical, Carter–Wegman)
hash31(a,b,x) = ((a*x + b) mod (2^31 - 1)) then mod w. Implemented with Mersenne-prime fast reduction:
  result = a*x + b; result = ((result >> 31) + result) & (2^31-1); // = result mod (2^31-1)
a,b random in [0, 2^31-1]. This is a 2-universal / pairwise-independent family (Carter–Wegman 1979).
4-wise (fourwise) = nested hash31 calls — NOT needed for CM (only AMS needs it).

## Canonical code structure (MassDal, Cormode 2003) → maps to scaffold
- CM_Init(width, depth, seed): pick hasha[j], hashb[j] per row; counts[depth][width].
- CM_Update(cm, item, diff): cm->count+=diff; for each row j: counts[j][hash31(...)%width] += diff.
- CM_PointEst(cm, q): min over rows of counts[j][hash31(...)%width].
- CM_PointMed: median variant for signed case.
- CM_InnerProd: per-row dot of the two count arrays, take min.
- CMH_*: hierarchical for range/HH/quantile.

## Evaluation settings (pre-method)
Streams: IP packet traces, query logs; metrics: per-item error vs ||a||_1, F_2 self-join error, space (KB),
update throughput (M updates/sec). Zipf streams with skew z. Yardstick baselines: AMS, Count Sketch.

## Design-decision → why
- min over rows (not avg/median): one-sided overestimate makes min the tightest valid upper bound; avg would
  include collision noise from every row; median is for the SIGNED case only.
- non-negative ⇒ Markov (first moment) ⇒ 1/eps and pairwise indep; vs Count Sketch's signs ⇒ variance ⇒ Chebyshev ⇒ 1/eps^2 + 4-wise-ish.
- w = e/eps: e is the Markov constant P(X>e·E)<1/e that minimizes total space wd for fixed accuracy (b=e optimum).
- d = ln(1/delta): rows are independent trials, failure (1/e)^d, set = delta.
- pairwise independence: only first moment used ⇒ 2-universal enough; cheap hash31.
- L1 guarantee (not L2): consequence of summing non-neg collision mass; good for skewed data where ||a||_1 dominated by heavy items.
