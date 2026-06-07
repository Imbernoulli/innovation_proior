# Synthesis — HyperLogLog discovery trace

## Pain point
Count distinct elements n in a huge stream (multiset M) in ONE pass with tiny fixed memory.
- Exact counting (hash set / sorted dedup) costs O(n) memory — linear, dead on arrival for n ~ 10^9 on a router or DB.
- Sampling fails because cardinality is sensitive to replication structure: a value seen 1M times vs once must count the same. Sampling a fraction f and scaling by 1/f estimates *occurrences* not *distinct*, and is wildly off for skewed data.
- Need: insensitive to replication (idempotent under repeats), one pass, ~constant work per element, sub-linear (ideally log log n) memory.

## Ancestors / lineage
1. **Morris 1977 approximate counting** — count up to n with log log n + O(1) bits by maintaining an exponent and incrementing probabilistically. Shows log log n memory is achievable for *counting occurrences*; not distinct-counting but plants the "log log" target.
2. **Flajolet–Martin 1985 (Probabilistic Counting / PCSA)** — THE foundational leap.
   - Hash each element to uniform bits. Observable: ρ(y) = position of leftmost/least-significant 1-bit (one plus run of leading 0s). Pattern 0^k 1 appears with prob 2^{-(k+1)}.
   - Maintain BITMAP[0..L-1]; set BITMAP[ρ(hash(x))]=1. After processing, BITMAP[i]≈1 for i ≲ log2 n, ≈0 for i ≳ log2 n, with a fringe.
   - R = position of leftmost ZERO in bitmap. E[R] ≈ log2(φ n), φ = 0.77351... (the "magic constant"), plus tiny periodic oscillation. σ(R) ≈ 1.12 bits → single bitmap is ±1 binary order of magnitude. Estimate 2^R/φ.
   - **Variance reduction**: average over m bitmaps. Direct averaging needs m hash functions (m× CPU). **Stochastic averaging**: use first part of hash to pick one of m bitmaps (a = h(x) mod m), rest for ρ. One hash, ~n/m per bucket. Estimate Z = (m/φ)·2^{S/m}, S = Σ R_j (arithmetic mean of the per-bucket leftmost-zero positions in the exponent → geometric mean of 2^{R_j}). Standard error 0.78/√m.
   - Memory: m bitmaps × L bits. For n up to 10^9, L=32, m=64 → 64×32 bits ≈ 256 bytes; 10% error.
3. **Durand–Flajolet 2003 (LogLog / SuperLogLog)** — shrink memory from a bitmap to a single small register per bucket.
   - Same observable ρ (leftmost 1-bit position) but keep only **M[j] = max ρ over bucket j** instead of a whole bitmap. Max leading-zero run ≈ log2(n/m). A register only needs to store a value up to ~log2(n/m)+const → log2 log2 n bits ≈ 5 bits ("small bytes"). Hence "LogLog": log log n memory.
   - Estimator: E = α_m · m · 2^{(1/m) Σ M[j]} — arithmetic mean of registers in exponent = **geometric mean** of the 2^{M[j]}. Constant α_m = (Γ(-1/m)(1-2^{1/m})/log2)^{-m}, → α̃_∞ ≈ 0.39701; standard error 1.30/√m.
   - SuperLogLog: truncate top ~30% of registers (restriction rule) before averaging to cut variance to ~1.05/√m, but not cleanly analyzable.
4. **Bar-Yossef / MinCount / Giroire** — order-statistics observables (min hash values), E[min]=1/(n+1); 1.00/√m. Alternative family; HLL paper notes it but the bit-pattern line is the one HLL continues.
5. **Chassaing–Gérin** — insight that harmonic-type means tame slow-decaying right tails; inspiration for HLL's harmonic mean. Also the 1/√m lower bound for order-statistics estimators (near-optimality).

## The HLL leap (2007)
Same observable as LogLog (max ρ per bucket = M[j]). ONLY change: the **evaluation function**. LogLog uses geometric mean (arithmetic mean of M[j] in exponent). HLL uses the **harmonic mean** of the 2^{M[j]}:
  Z = (Σ_j 2^{-M[j]})^{-1},   E = α_m · m^2 · Z = α_m m^2 / Σ_j 2^{-M[j]}.
Why harmonic: 2^{M[j]} is ~ geometrically distributed with a heavy/slow-decaying right tail; a bucket that happens to get a freak long leading-zero run produces a huge 2^{M[j]} that an *arithmetic* mean lets dominate (high variance). Harmonic mean weights by 1/2^{M[j]}, so freak-large registers contribute ~0 — it tames the right tail. Cuts std error from 1.30/√m to 1.04/√m.
- Bias correction constant α_m = (m ∫_0^∞ (log2((2+u)/(1+u)))^m du)^{-1}; asymptotically α_∞ = 1/(2 log 2) ≈ 0.72134. Practical values: α_16=0.673, α_32=0.697, α_64=0.709, α_m=0.7213/(1+1.079/m) for m≥128.
- Standard error β_m/√m, β_∞ = √(3 log 2 − 1) ≈ 1.03896 ≈ 1.04.
- Memory: m registers of ≤ log2 log2 N + O(1) bits = "5-bit short bytes". m=2048, 32-bit hash → 1.5 kB → 2% error up to 10^9. Near-optimal: Ω(log log N) bits is an information-theoretic floor (cardinalities on exponential scale), and 1/√m is the order-statistics accuracy floor.

## Corrections (Fig 3, 2007 program — for real code)
- Initialize registers to 0 (not −∞) so usable even when n is a small multiple of m.
- Raw E = α_m m^2 / Σ 2^{-M[j]}.
- **Small range** (E ≤ 5m/2): nonlinear distortion (empty buckets, coupon-collector). Let V = # registers still 0. If V≠0 use **linear counting**: E* = m·log(m/V) (balls-in-bins: empty bins ≈ m·e^{-n/m} ⇒ n ≈ m·log(m/V)). Else keep E.
- **Intermediate** (5m/2 < E ≤ (1/30)2^32): E* = E (no correction).
- **Large range** (E > (1/30)2^32): hash collisions near 2^32 saturate; invert E = 2^32(1 − e^{−n/2^32}) ⇒ E* = −2^32 log(1 − E/2^32).
- σ ≈ 1.04/√m; estimates within σ,2σ,3σ of truth in 65/95/99% of cases.

## ρ / register indexing (code)
64-bit hash x. j = first p bits (here implementation uses low p bits: j = x & (m−1)); w = remaining bits; ρ(w) = (64−p) − bitlength(w) + 1 = 1 + leading-zero run. M[j] = max(M[j], ρ(w)). m = 2^p, p = ceil(log2((1.04/err)^2)).

## Design-decision → why
- ρ = leftmost-1 position (not rightmost-1): rightmost gives flatter distribution, worse std error (FM note). Leftmost run of zeros directly tracks log2 of count.
- Keep max ρ not whole bitmap: bitmap is L bits/bucket; max is one ⌈log2 log2 n⌉-bit register. log→loglog memory.
- Stochastic averaging not m hash functions: one hash, constant work/element; no need for an unknown large family of independent hashes.
- Harmonic over geometric/arithmetic mean: tames slow-decaying right tail of 2^M (variance reduction), 1.04 vs 1.30 vs blow-up.
- α_m bias correction: the raw m^2/Σ2^{-M} is biased by a multiplicative constant; α_m removes it (derived via Mellin/Poissonization, but in-frame derive the *need* and the small-m empirical values).
- Linear counting at small n: coupon-collector — when n≲m log m many buckets empty, raw estimator collapses (α_m m ≈ 0.7m when n=0). Empty-bin count is itself a clean estimator there.
- Large-range: 32-bit hash space saturates by collisions; invert collision model.
- 5-bit registers: register value ≤ log2 log2 N + O(1); for N≤10^9, ≤ ~5.

## In-frame rules
- No mention of any source paper as artifact. May name FM/PCSA, LogLog, Morris as prior-art ancestors by author/year (they ARE the lineage). The TARGET (HyperLogLog) is the thing being invented — name it only in answer.md as the result, not as a citation.
