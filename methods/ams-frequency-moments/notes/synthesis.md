# Synthesis — AMS Frequency Moments (F_2 in polylog space)

## Primary source (read in full)
- `refs/alon_matias_szegedy_amsz4.pdf` / `.txt` — Alon, Matias, Szegedy, "The space complexity of approximating the frequency moments" (authors' own corrected copy of STOC 1996 / JCSS 58(1):137–147, 1999). 19 pages of extracted text, read end to end.

## Antecedents (read)
- `refs/flajolet_martin_1985_probabilistic_counting.pdf` — Flajolet & Martin 1985, "Probabilistic Counting Algorithms for Data Base Applications," JCSS 31:182–209. F_0 (distinct elements) in O(log n) bits via hashing + rank-of-lowest-set-bit / BITMAP, output ~2^R. Assumes ideal hash. AMS [ref 7] reacts to this: replaces ideal hash with explicit linear hash (Prop 2.3).
- `refs/morris_1978_counting_small_registers.pdf` — Morris 1978, "Counting large numbers of events in small registers," CACM 21(10):840–842. Approximate counter: store v ≈ log of count; on event increment v with probability ~Δ (≈2^{-v}); estimate ≈ 2^v − 1. O(log log m) bits for F_1. AMS [ref 15] uses this both as an antecedent and as a subroutine (to store r / the sum Z with O(log log m) bits for very large m).

## Analyses (read)
- `refs/chakrabarti_data_stream_algorithms.{pdf,txt}` — Amit Chakrabarti, "Data Stream Algorithms" lecture notes (Dartmouth). Units 6 (basic AMS estimator for F_k), 7 (Tug-of-War sketch for F_2), 4.4 (median-of-means). Cleanest re-derivation of E[Z^2]=F_2, Var(Z^2)≤2F_2^2, median-of-means.
- `refs/utexas_lec6.{pdf,txt}` — UT Austin CS395T (Eric Price), Lecture 6. Connects AMS sketch to JL / dimension reduction; states the 4-wise-independence requirement and the variance bound.
- `refs/analysis_thejaswi_westrup_aalto_2016.{pdf,txt}` — Aalto term-project report applying F_0 and F_2 estimators to 8.8GB Twitter data; practical hash-family discussion.

## The problem (in-frame)
- Stream A=(a_1,...,a_m), a_j ∈ {1..n}. m_i = count of i. F_k = Σ_i m_i^k. F_0 = #distinct, F_1 = m, F_2 = Σ m_i^2 = repeat rate / self-join size / Gini homogeneity.
- Exact F_k needs a histogram: Ω(n log m) bits. External memory => expensive. Want one pass, sublinear (ideally polylog) space, randomized (ε,δ)-approximation.
- F_2 motivation: self-join size r ⋈ r; database skew; query-optimization. Maintaining F_2 online as records insert is valuable.

## The F_2 leap (heart) — re-derive
1. Want Σ m_i^2 in one pass, polylog space. Can't store the m_i (that's Ω(n)).
2. Linear-sketch idea: keep a single scalar Z that is a linear function of the m_i, so it updates by += on each token and is mergeable. Try Z = Σ_i ε_i m_i with ε_i random signs ±1.
3. ε_i = h(i), h:[n]→{−1,1}. On token j: Z += h(j). |Z| ≤ m so O(log m) bits; h stored in O(log n).
4. Estimator X = Z^2. E[Z^2] = E[(Σ ε_i m_i)^2] = Σ m_i^2 E[ε_i^2] + Σ_{i≠j} m_i m_j E[ε_i ε_j]. ε_i^2=1; if pairwise independent and mean-zero, E[ε_i ε_j]=0 (i≠j). So E[X]=Σ m_i^2 = F_2. UNBIASED with only pairwise independence + mean zero.
5. Variance. Var(X)=E[Z^4]−F_2^2. E[Z^4]=Σ_{i,j,k,l} m_i m_j m_k m_l E[ε_i ε_j ε_k ε_l]. A 4-tuple's expectation is nonzero only when the indices pair up (each ε appears an even number of times), because any index appearing an odd number of times factors out a mean-zero E[ε]=0 — and to *guarantee* the factorization for distinct indices we need independence up to 4 of them: **4-wise independence**. Surviving terms: all four equal (Σ m_i^4 = F_4, coeff 1) and two distinct pairs (i,i,j,j): coefficient = number of orderings of (i,i,j,j) = 4!/(2!2!) = 6, giving 6 Σ_{i<j} m_i^2 m_j^2. So E[Z^4] = F_4 + 6 Σ_{i<j} m_i^2 m_j^2.
6. Var(X) = F_4 + 6 Σ_{i<j} m_i^2 m_j^2 − F_2^2. Note F_2^2 = (Σ m_i^2)^2 = Σ m_i^4 + 2 Σ_{i<j} m_i^2 m_j^2 = F_4 + 2 Σ_{i<j} m_i^2 m_j^2. So Var(X) = (F_4 + 6S) − (F_4 + 2S) = 4S where S = Σ_{i<j} m_i^2 m_j^2. And 2S ≤ F_2^2 (since F_2^2 = F_4 + 2S ≥ 2S). Hence Var(X) = 4S = 2·(2S) ≤ 2 F_2^2.  ✓ (AMS Thm 2.2; Chakrabarti 7.1.1: Var Z^2 = F_4 − F_2^2 + 6S = F_4 − F_2^2 + 3(F_2^2 − F_4) ≤ 2F_2^2.)
7. Why 4-wise and not full independence: full independence over [n] would cost Ω(n) random bits to store. 4-wise is *exactly* what the 4th-moment computation touches (the ε_iε_jε_kε_l terms), and small 4-wise-independent families exist: degree-3 polynomial over GF(p) (h(i)=sign of a cubic), or BCH-code / orthogonal-array-of-strength-4 construction — O(log n) seed bits, each ε_i computable in O(log n) space. AMS uses the BCH construction (ref [1]); textbook construction is the GF(p) cubic.
8. Why pairwise was enough for unbiasedness but not variance: E[X] only touches pairs ε_iε_j (2-wise); Var touches quadruples (4-wise). So the design requirement is set by the variance step, not the mean.
9. (ε,δ) amplification — median-of-means. One X has relative std ~√(Var/E^2)=√(2F_2^2)/F_2=√2, too big. Average s_1 = O(1/ε^2) iid copies → Y_i with Var(Y_i)=Var(X)/s_1 ≤ 2F_2^2/s_1; Chebyshev: P(|Y_i−F_2|>εF_2) ≤ Var(Y_i)/(εF_2)^2 ≤ 2/(s_1 ε^2) ≤ 1/8 for s_1 = 16/ε^2. Then take median of s_2 = O(log 1/δ) such averages; Chernoff: P(median off by >εF_2) ≤ δ. AMS: s_1=16/λ^2, s_2=2 log(1/δ). Total space O(ε^{-2} log δ^{-1} (log n + log m)).

## General F_k estimator (basic AMS, Thm 2.1)
- Pick random position p∈[m] (reservoir sampling: keep a with prob 1/m as m grows). Let l=a_p, r = #occurrences of l from p to end. X = m(r^k − (r−1)^k).
- E[X] = (m/m)·Σ_i Σ_{c=1}^{m_i} (c^k−(c−1)^k) ... = Σ_i m_i^k = F_k (telescoping per value).
- E[X^2] ≤ m·Σ_i Σ_c (c^k−(c−1)^k)^2. Using a^k−b^k ≤ (a−b) k a^{k−1}: (c^k−(c−1)^k) ≤ k c^{k−1}, so the inner sum ≤ k Σ_c c^{k-1}(c^k−(c−1)^k) telescopes ≤ k m_i^{2k−1}. So E[X^2] ≤ k m Σ_i m_i^{2k−1} = k F_1 F_{2k−1}.
- Fact: (Σ m_i)(Σ m_i^{2k−1}) ≤ n^{1−1/k} (Σ m_i^k)^2, i.e. F_1 F_{2k−1} ≤ n^{1−1/k} F_k^2. So Var(X) ≤ E[X^2] ≤ k n^{1−1/k} F_k^2.
- s_1 = 8 k n^{1−1/k}/λ^2, median-of-means ⇒ space O(k λ^{-2} log(1/δ) n^{1−1/k}(log n + log m)). For k=2 this is √n; the tug-of-war improves k=2 to log.

## F_0 (Prop 2.3, AMS's fix of Flajolet–Martin)
- Linear hash f(x)=a·x+b over GF(2^d), 2^d>n. r(z)=#trailing zero bits. R=max over stream of r(f(a_i)). Output Y=2^R. Pairwise independence of f suffices; for 2^r>cF_0, P(some token reaches rank r)<1/c; for c2^r<F_0, P(none reaches r)<1/c (Chebyshev). So Y within factor c of F_0 w.p. ≥1−2/c. O(log n) bits.

## Lower bounds (Section 3)
- F_∞* (max m_i): reduce DISJOINTNESS (Kalyanasundaram–Schnitger Ω(n)); a streaming approx of F_∞* gives a one-way protocol ⇒ Ω(n) space. (F_∞*=1 disjoint, 2 intersecting.)
- F_k, k>5: multiparty DIS(s,t) game, s players, t-subsets; disjoint ⇒ F_k=st, uniquely-intersecting ⇒ F_k ≈ (3/2)n; gap distinguishes. Razborov-style box/partition argument (Lemma 3.4–3.6, entropy) ⇒ protocol Ω(t/s^3) ⇒ space M ≥ Ω(n^{1−5/k}). s=n^{1/k}, t=Θ(n^{1−1/k}).
- Deterministic: any k≠1 needs Ω(n) (equality / code-family pigeonhole, Prop 3.7). Randomized *exact*: Ω(n) for k≠1 (Prop 3.8). So both randomness AND approximation are necessary.
- Tight lower bounds F_0 Ω(log n), F_1 Ω(log log m), F_2 Ω(log n + log log m) (Prop 3.9).

## Walls / self-corrections to live in reasoning.md
- Wall: histogram is Ω(n). Patch: don't store m_i; store one linear scalar.
- Wall: a plain sum Σ m_i collapses everything (gives F_1, not F_2). Patch: randomize signs so squaring resurrects the squares.
- Wall: E[Z^2] has cross terms — only die if E[ε_iε_j]=0 ⇒ need mean-zero + pairwise independence.
- Wall: variance — E[Z^4] cross terms only die under 4-wise independence; full independence too expensive ⇒ realize 4-wise suffices AND is cheaply constructible (cubic over GF(p) / BCH).
- Wall: single estimator's variance (√2 relative) too big for (ε,δ) ⇒ average to cut variance, but averaging alone needs O(1/(ε^2δ)) copies (Chebyshev on the average) ⇒ instead median-of-means: O(1/ε^2) per group, O(log 1/δ) groups, exponential δ via Chernoff.

## Code grounding
- Tug-of-War: z += h(token); output z^2; t1×t2 table of independent (h, z); average rows, median of column-means.
- 4-wise independent ±1 hash: standard construction = degree-3 polynomial over prime field GF(p), p>n: a0+a1 x+a2 x^2+a3 x^3 mod p, map to sign by (value mod 2 → ±1) [this is the textbook 4-universal → ±1]. Seed = (a0,a1,a2,a3). AMS's own construction is BCH/orthogonal-array; the cubic is the standard teaching implementation and is genuinely 4-wise independent. Code mirrors Chakrabarti Algorithm 11 + median-of-means (Lemma 4.4.1).
