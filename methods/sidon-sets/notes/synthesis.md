# Synthesis — Sidon sets and difference-set constructions

## The task / pain point
Build A ⊆ {1,...,n} with all pairwise SUMS distinct (B2 / Sidon set), as large as possible.
Equivalent: all pairwise DIFFERENCES distinct (a+b=c+d ⇔ a−c=d−b). Want |A| ≈ √n.
Anti-pattern = brute-force/backtracking search (Golomb-ruler search). Want PRINCIPLED algebraic constructions.

## Upper bound (Erdős–Turán 1941, Lindström refinement) — double counting
- Crude: a Sidon set of size r has C(r,2) distinct positive differences, all in {1,...,d}, so C(r,2)≤d ⇒ r ≤ √(2d) ≈ 1.41√d.
- Refined (gives √d + d^{1/4}+1): order differences. For A = a_1<...<a_r, "order ν" difference = a_{i+ν}−a_i.
  Σ_ν = Σ_{i=1}^{r−ν}(a_{i+ν}−a_i). Split into ν telescoping chains, each ≤ d, so Σ_ν ≤ νd.
  Sum over ν=1..m: Σ_1+...+Σ_m ≤ (1+2+...+m)d = ½m(m+1)d.    (UPPER, eq A)
  Count of differences of order ≤ m: (r−1)+(r−2)+...+(r−m) = mr − ½m(m+1) = ms, with s = r − ½(m+1).
  All distinct positive integers ⇒ their sum ≥ 1+2+...+ms = ½ms(ms+1) > ½ m²s².  (LOWER, eq B)
  Combine: ½ m²s² < ½ m(m+1)d ⇒ s² < (1+1/m)d ⇒ s < √d·√(1+1/m) < √d(1+1/(2m)).
  r = s + ½(m+1) < ½(m+1) + √d(1+1/(2m)). Optimize m = ⌊d^{1/4}⌋:
  r < √d + d^{1/4} + 1.   [VERIFIED algebra against Golomb PDF p.22-23]

## Construction 1 — Erdős–Turán quadratic-residue (the elementary one)
p odd prime. Let (k²)_p ∈ {0,...,p−1} be k² mod p. Set
   s_k = 2p·k + (k²)_p,  k = 0,...,p−1.
- |A| = p, all in [0, 2p²), so n ≈ 2p² ⇒ |A| ≈ √(n/2).
- PROOF it's Sidon: suppose s_i+s_j = s_k+s_l.
  2p(i+j) + (i²)_p+(j²)_p = 2p(k+l) + (k²)_p+(l²)_p.
  ⇒ 2p[(i+j)−(k+l)] = [(k²)_p+(l²)_p] − [(i²)_p+(j²)_p].
  RHS is a difference of two sums each in [0,2p−2], so |RHS| < 2p. LHS is multiple of 2p. Both ⇒ 0.
  So (1) i+j = k+l, and (2) (i²)_p+(j²)_p = (k²)_p+(l²)_p ⇒ i²+j² ≡ k²+l² (mod p).
  From (1): i+j ≡ k+l. Then (i+j)² = (k+l)² mod p combined with i²+j²≡k²+l² gives 2ij ≡ 2kl,
  and since p odd, ij ≡ kl (mod p). So i,j and k,l have equal sum and equal product mod p ⇒ same roots
  of X²−(i+j)X+ij ≡ 0 ⇒ {i,j}≡{k,l} (mod p). Indices in {0,...,p−1} ⇒ {i,j}={k,l}. ∎
  [VERIFIED in code: erdos_turan(p) Sidon for p up to 31.]
  Note: well-definedness of (k²)_p picking the representative in [0,p−1] is what makes |RHS|<2p work;
  the 2p "stride" cleanly separates the coarse part (k index) from the fine part (residue).

## Construction 2 — Singer perfect difference set (PG(2,q))
q prime power, v = q²+q+1. There exist q+1 integers d_0,...,d_q (mod v) whose q²+q pairwise differences
hit EVERY nonzero residue mod v exactly once (a "perfect difference set"). This is the strongest:
density q+1 in {1,...,v}≈ √v, and not only distinct but perfect.
- Geometric origin: PG(2,q) has v points, v lines, each line q+1 points, each pair of points on exactly one
  line (a 2-(v,q+1,1) design). Singer (1938): there is a collineation (Singer cycle) of order v cyclically
  permuting all points: P_i = σ^i(P_0). A line's point-indices {d_0,...,d_q} then form a perfect difference set:
  for any nonzero t, P_a and P_{a+t}=σ^t(P_a) lie on a unique common line, i.e. t = d_i − d_j uniquely.
- Algebraic realization: G = F_{q³}*/F_q* is cyclic of order v. The "line" = projectivization of the
  trace-zero hyperplane H = {x ∈ F_{q³} : Tr(x)=x+x^q+x^{q²}=0}, |H\{0}| = q²−1, collapsing by F_q* gives
  (q²−1)/(q−1)=q+1 cosets. With θ a generator of F_{q³}*, D = { i mod v : Tr(θ^i)=0 }.
  [VERIFIED in code: q=2 → {1,2,4} mod 7 perfect; q=3 → {0,7,8,11} mod 13 perfect.]

## Construction 3 — Bose (B2 in Z_{q²−1}, the affine cousin)
q prime power, θ primitive in F_{q²}. D = { a ∈ [1,q²−1] : θ^a − θ ∈ F_q }. |D| = q, Sidon mod q²−1.
- PROOF (Bose–Chowla): if a+b ≡ c+d (mod q²−1) with all four in D, write θ^a=θ+α, θ^b=θ+β,
  θ^c=θ+γ, θ^d=θ+δ, with α,β,γ,δ ∈ F_q. θ^{a+b}=θ^{c+d} ⇒ (θ+α)(θ+β)=(θ+γ)(θ+δ):
  θ² + (α+β)θ + αβ = θ² + (γ+δ)θ + γδ. Since {1,θ} is an F_q-basis of F_{q²} (θ ∉ F_q), match coords:
  α+β = γ+δ and αβ = γδ ⇒ {α,β}={γ,δ} ⇒ {a,b}={c,d}. ∎
  [VERIFIED q=2,3,5,11; q=11 gives 11 marks mod 120.]

## Why three constructions, what's the "leap"
- The methodological leap: stop SEARCHING for Sidon sets; instead realize that the Sidon (distinct-sums)
  condition is exactly the "no nontrivial additive collision" condition, which is what an injective
  homomorphism into a multiplicative group buys you. Map indices → exponents of a primitive element so that
  additive collisions a+b=c+d become MULTIPLICATIVE collisions θ^aθ^b=θ^cθ^d, then a low-degree algebraic
  identity (uniqueness of quadratic factorization / 2-element symmetric functions) forces {a,b}={c,d}.
  Erdős–Turán's 2pk+(k²)_p is the "poor man's" version of the same idea using k²: the residue (k²)_p plays
  the role the multiplicative structure plays in Bose/Singer; the proof reduces to: equal sum + equal
  sum-of-squares ⇒ equal product ⇒ same pair.
- Singer adds the geometric PG(2,q) lens that upgrades "distinct" to "perfect" (every difference once).
- Matching: upper bound √n + O(n^{1/4}); Singer gives q+1 ≈ √n in {1,...,q²+q+1}; Bose gives q ≈ √n in
  {1,...,q²−1}; ET gives p ≈ √(n/2). All hit the √n order.

## Code grounding
- Bose pseudocode from Golomb PDF p.61: iterate ζ=θ^n, collect n where ζ−θ ∈ GF(q); sorted output.
- My verify_constructions.py / verify_singer_bose.py implement and CHECK all three + perfect-difference.
- Final answer code: pure-Python GF(p^d) arithmetic + the three constructors + is_sidon verifier.

## Sources
- Golomb/Sidon analysis (Toronto, Apostolopoulos): refs/golomb_main.txt — upper bound proof, ET, Singer, Bose, Ruzsa, pseudocode.
- Bose–Chowla argument: arxiv 2104.12711.
- Upper bound: arxiv 2103.15850 (Lindström-style).
- ET quadratic construction: Pohoata blog + arxiv 2207.07800 ("On the size of finite Sidon sets").
- Singer/trace-zero/PG(2,q): Szabó "Singer difference sets and the projective norm graph" (FU Berlin / arxiv 1908.05591); search summaries.

## Uncertainty flags
- The "order ν" telescoping bound: ν chains each ≤ d because telescoping sum a_{kν+r}−...−a_r ≤ a_max−a_min ≤ d−1<d. Solid.
- Singer index set depends on choice of generator/representatives; the SET is canonical up to translation/multiplier. My code's exact D varies with modpoly but always perfect — consistent.
- Bose "q² vs q²−1" largest element: marks live in {0,...,q²−2}; book says "<q²−1". Fine.
