# Synthesis: Heilbronn Triangle Problem

## Problem
Δ(n) = sup over n-point sets P ⊂ [0,1]² of min over triples of area(triangle). Want asymptotics.
Trivial: Δ ≤ 1/(2n) by triangulation/pigeonhole (n−2 disjoint triangles in unit square). Erdős/probabilistic Δ ≥ Ω(1/n²). Heilbronn conjectured Δ = O(1/n²) (i.e. ≍ 1/n²).

## Two principled targets

### LOWER BOUND — probabilistic placement + deletion (KPS 1982), Δ = Ω(log n / n²)
Anti-pattern: numeric/simulated-annealing point coordinates. Principled method:

**Basic deletion (Ω(1/n²)), grounded in Zhao notes §3.2:**
- Place 2n points uniform in [0,1]².
- Key probability: fix p,q. dist of pair |pq| has density ≤ 2πx dx (annulus area). Given |pq|, P_r(area(pqr) ≤ ε) = P(dist(r, line pq) ≤ 2ε/|pq|) ≲ (2ε/|pq|)·(diam) ≲ ε/|pq| (strip of width 2ε/|pq|, length ≤ √2).
  Integrate: P_{p,q,r}(area ≤ ε) ≲ ∫₀^{√2} (ε/x)·2πx dx ≍ ε.  ← linear in ε.
- E[X] = E[# triples area ≤ ε] = O(ε n³). Choose ε = c/n² so E[X] ≤ n.
- Delete one point per bad triangle: ≥ 2n − X ≥ n points remain, no triangle ≤ ε=c/n². → Δ ≥ Ω(1/n²).
- Erdős algebraic alt: {(x, x² mod p)/p}: parabola hits each line ≤2 pts, dilate to lattice, Pick → area ≥ 1/2p² ≥ 1/8n². Also Ω(1/n²).

**Why the naive deletion stalls at 1/n²:** deleting one vertex per bad edge kills up to X ≈ n points — it only "pays" 1 surviving point per bad triple. To push ε up to (log n)/n² you'd have E[X] = O(εn³) = O(n log n) ≫ n bad triples; naive deletion removes everything. Need to delete far fewer points than there are bad triples.

**The leap (KPS / AKPSS independent set):** View the bad-area triples as edges of a 3-uniform hypergraph H on the N placed points. We don't need to delete one-per-edge; we need a large INDEPENDENT SET (contains no full bad triple). For a *sparse/uncrowded* hypergraph the independence number is much larger than naive deletion gives.

Ajtai–Komlós–Pintz–Spencer–Szemerédi (Extremal Uncrowded Hypergraphs, JCTA 1982): a k-uniform UNCROWDED hypergraph (no 2-,3-,4-cycles) on N vertices with average degree t^{k−1} has
   α(H) ≥ C_k (N/t)(ln t)^{1/(k−1)}.
For triangles k=3: α(H) ≥ C₃ (N/t)(ln t)^{1/2}, with t² = 3|E|/N the avg degree.
The (ln t)^{1/2} is EXACTLY the log gain. Naive deletion gives only ≈ N/t² · (something) i.e. loses the ln factor; the AKPSS semi-random "nibble"/random-greedy argument harvests (ln t)^{1/2}.

**Uncrowdedness must be arranged:** the placed-point bad hypergraph is not automatically uncrowded — need to delete points in the few 2-,3-,4-cycles (two bad triangles sharing 2 pts; three/four bad triangles forming short cycles). KPS / Wikipedia steps: place n^{1+ε} points; remove unexpectedly-close pairs (controls a 2-cycle = two near-degenerate triangles on a shared edge); show # of cycles of 2,3,4 bad triangles is sublinear (o(N)); delete those points; the rest is uncrowded; apply AKPSS to get the independent set. Second-moment / expectation bounds show the cycle counts are small.

**Parameter scaling (my derivation, to state in reasoning):**
- N placed points, threshold ε. |E| ≈ ε N³ ⇒ avg degree t² = 3|E|/N ≈ ε N². So t ≈ N√ε.
- α ≈ (N/t)(ln t)^{1/2} = (1/√ε)(ln t)^{1/2}.
- want α = n at the largest ε. With N ≍ n: t ≈ n√ε, set ε = c(ln n)/n² ⇒ √ε = √(c ln n)/n ⇒ 1/√ε = n/√(c ln n); t ≈ n·√(c ln n)/n = √(c ln n) ⇒ ln t ≈ ½ ln ln n... gives α ≈ n/√(c ln n) · (½ ln ln n)^{1/2} — too small.
  The honest accounting: take N a bit larger so t is a genuine power of n. Standard KPS choice: N = n^{1+δ}, ε ≍ (log n)/n² ⇒ t² ≈ ε N² = n^{2δ} log n ⇒ t ≈ n^δ, ln t ≈ δ log n. Then α ≈ (N/t)(ln t)^{1/2} = (n^{1+δ}/n^δ)(δ log n)^{1/2} ≍ n (log n)^{1/2}. After removing o(N) cycle-points, still ≳ n√(log n) ≥ n. So at area threshold ε ≍ (log n)/n² we retain ≥ n points. → Δ ≥ Ω(log n / n²). ✓
  (The (log n)^{1/2} slack over n is what lets the threshold be pushed all the way to log n/n²; equivalently solving α = n gives the maximal ε ≍ log n/n².)
- Derandomized to poly-time by Bertram-Kretzberg–Hofmeister–Lefmann (1997/2000) via deterministic independent-set-in-uncrowded-hypergraph algorithm. Constructive code: random placement + greedy/semi-random independent set is the real algorithm.

### UPPER BOUND — strip/incidence framework (Roth 1951 density increment; Schmidt 1972 energy; Roth 1972 analytic two-scale; KPS 1981 optimize to 8/7; CPZ 2023 high-low + projection)
Grounded in CPZ §1–3 (arXiv 2305.18253).

**Strip reformulation (the unifying object):** for pair τ={x,y}, T_τ(w) = strip of width w about line ℓ_τ. area(x,y,z) < ∆ ⟺ z ∈ T_τ(4∆/d(τ)), d(τ)=|xy|. If ∆ = min area, then T_τ(4∆/d(τ)) ∩ P = {x,y} for every pair — strips are "empty". If ∆ is large, strips are wide yet empty → tension.

**Schmidt energy (cleanest, Δ ≤ n^{-1}(log n)^{-1/2}):**
Weight each strip edge by w(T)=4∆/d(τ). Weighted degree of any x: Σ_T w(T)1_T(x) ≲ 1 (a point lies in few empty strips, weighted). ⇒ weighted incidences I_w(S,L)=∫_S Σ w(T)1_T ≲ 1. But also I_w = Σ_τ (4∆/d(τ))·area(strip∩S) ≳ Σ_τ ∆/d(τ)·... actually = Σ (4∆/d(τ))∫1_T ≳ Σ ∆/d(τ) · (d(τ)) → ∆² Σ 1/d(τ)². Riesz 2-energy Σ_{τ} 1/d(τ)² ≳ n² log n (∼log n distance scales, each ≳ n²). So ∆²·n² log n ≲ 1 ⇒ ∆ ≲ n^{-1}(log n)^{-1/2}.

**Roth analytic / density increment (polynomial improvement, ∆ ≲ n^{-1-µ}):**
restrict to pairs with d(τ) ≤ u and similar slope → strips ≈ disjoint; complement has higher point density; iterate (1951 → o(1/n), explicitly n^{-1}(log log n)^{-1/2}). 1972 analytic version: count smoothed incidences I(w;P,L)=Σ η(w^{-1}d(p,ℓ)) between points and the lines L of short pairs. ∆ ≤ u·d(P,L). Two steps: (A) initial estimate I(w_i) ≳ w_i|P||L| (point sets without small triangles look pseudo-random at coarse scale w_i); (B) inductive step: incidence density barely changes from w_i to small w_f, via quasi-orthogonality of strip indicator differences ϕ_ℓ = (1/w_i)1_{T(w_i)} − (1/w_f)1_{T(w_f)} (Selberg/Bessel inequality). At w_f ≪ w_i still many incidences ⇒ ∆ ≲ u w_f. Optimize: Roth µ = 1−√(4/5) ≈ .1056, later (9−√65)/8 ≈ .1172. KPS optimize → µ = 1/7 ⇒ ∆ ≤ exp(c√log n)/n^{8/7}.

**CPZ 2023 (break 8/7):** replace Selberg machinery by the high-low method (Guth–Solomon–Wang 2017): renormalized incidence counts change slowly across scales unless points/lines concentrate; |I(w_i)/(w_i|P||L|) − I(w_f)/(w_f|P||L|)| ≲ (M_P(w_f²)M_L(w_i×1)/(|P||L|))^{1/2} w_f^{-3/2}. Worst case = clustered-at-small-scale-but-spread-globally (fractal). Control via projection theory: dim of direction set S(X) (Marstrand 1954; refined Orponen–Shmerkin–Wang 2022) — discretized sum-product / Bourgain. Yields ∆ ≤ n^{-8/7-1/2000}; homogeneous sets n^{-7/6+o(1)}.

## Design-decision → why table
- Place 2n (not n) points / N=n^{1+δ}: need slack to delete bad-config points and still have ≥ n. Basic version: 2× slack. Log version: poly slack so independent set ≥ n.
- Threshold ε linear-in-ε probability bound: comes from strip geometry, the SAME strip object as upper bound. ε=c/n² (basic) vs ε≍log n/n² (KPS).
- Delete one-per-triangle (basic) vs independent set in uncrowded hypergraph (KPS): one-per-triangle wastes; independent set in sparse 3-graph keeps (ln t)^{1/2} more → the log gain.
- Remove close pairs / short cycles first: to make hypergraph uncrowded so AKPSS applies; second-moment shows # such cycles = o(N).
- Upper bound: strip width 4∆/d(τ): exact locus of area<∆. weight by width: makes weighted degree O(1) (Schmidt). restrict to short similar-slope pairs: makes strips disjoint (Roth density increment). smoothed incidences + quasi-orthogonal scale differences: lets you compare two scales analytically (Roth analytic). high-low + projection: finer scale access, beats 8/7 (CPZ).
- Why 8/7 specifically: optimization of the two-scale incidence exponents (u, w_f) in Roth's method; KPS found the optimum at µ=1/7.

## Sources read this run
- Zhao, Probabilistic Methods notes §3.2 (basic deletion, exact prob bound) — yufeizhao.com/pm/probmethod_notes.pdf
- Cohen–Pohoata–Zakharov arXiv 2305.18253 §1–3 (full strip/incidence framework, Roth/Schmidt/KPS history with exact exponents, high-low, projection)
- Wikipedia Heilbronn triangle problem (history, KPS lower-bound steps, deterministic algorithm)
- AKPSS independence theorem: arXiv 1602.03569 Thm 4 (α ≥ C_k (N/t)(ln t)^{1/(k−1)})
- Quanta 2023 (strip intuition, CPZ narrative)
- Bertram-Kretzberg–Hofmeister–Lefmann (derandomization, constructive poly-time)

## Code grounding
Real algorithm = random placement + (remove crowded configs) + independent set in the bad-triangle hypergraph (greedy/semi-random). I'll write numpy: place points, build bad-triple hypergraph at threshold ε, make it uncrowded by deleting low-multiplicity-cycle points, then random-greedy independent set. This is the honest constructive form (matches BKHL derandomization in spirit). Basic version: place 2n, delete one per bad triple → simplest faithful code.
