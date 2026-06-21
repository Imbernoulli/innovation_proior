# The Second Moment Method

**Problem.** For a nonnegative integer-valued counting variable X = X(n) (number of copies of a
configuration in a random structure), the first moment gives only one direction: E[X] → 0 ⇒ X = 0
almost always. But E[X] → ∞ does **not** imply Pr[X > 0] → 1 — a variable large on a rare event has a
huge mean yet is usually 0. The second moment method supplies the missing direction.

**Key idea.** Control the variance. If X is large in mean *and* tightly concentrated (small variance
relative to the mean squared), it cannot also be 0 most of the time. Chebyshev converts this into an
existence proof; Paley–Zygmund gives a constant-probability version from the same two moments.

## Core theorems

**Chebyshev.** For X with mean µ, variance σ², and any λ > 0: Pr[|X − µ| ≥ λσ] ≤ 1/λ².
(*Proof:* Markov on (X−µ)²: Pr(|X−µ|≥α) = Pr((X−µ)²≥α²) ≤ E[(X−µ)²]/α² = σ²/α²; set α = λσ.)

**Second-moment existence.** Pr[X = 0] ≤ Var[X]/E[X]².
(*Proof:* X = 0 ⇒ |X − µ| = µ, so Pr[X=0] ≤ Pr[|X−µ| ≥ µ] ≤ σ²/µ² by Chebyshev.)

> **Corollary.** If Var[X] = o(E[X]²) then X > 0 almost always. The same bound on |X−µ| ≥ εE[X] gives
> Pr[|X−E[X]| ≥ εE[X]] ≤ Var[X]/(ε²E[X]²), hence Var[X] = o(E[X]²) ⇒ **X ∼ E[X]** almost always.

**Variance of a sum of indicators.** For X = Σᵢ 1_{Aᵢ}, pᵢ = Pr[Aᵢ]:
Var[X] = Σᵢ Var[Xᵢ] + Σ_{i≠j} Cov[Xᵢ,Xⱼ]. Since Var[Xᵢ] = pᵢ(1−pᵢ) ≤ E[Xᵢ] and Cov = 0 for
independent pairs, writing i∼j for dependent pairs and Δ = Σ_{i∼j} Pr[Aᵢ∧Aⱼ]:

  **Var[X] ≤ E[X] + Δ.**   So *E[X] → ∞ and Δ = o(E[X]²) ⇒ X > 0 and X ∼ E[X] almost always.*

**Symmetric form.** If the events are symmetric, Δ = Δ*·E[X] with Δ* = Σ_{j∼i} Pr[Aⱼ|Aᵢ] (any fixed i):

  *E[X] → ∞ and **Δ* = o(E[X])** ⇒ X > 0 and X ∼ E[X] almost always.*

Reading: conditioning on one copy being present must not substantially raise the expected count of the
others.

**Paley–Zygmund (constant-probability refinement).** For Z ≥ 0 and t ∈ [0,1]:

  **Pr[Z > tE[Z]] ≥ (1−t)² E[Z]²/E[Z²].**

(*Proof:* E[Z] = E[Z·1_{Z≤tE[Z]}] + E[Z·1_{Z>tE[Z]}] ≤ tE[Z] + √(E[Z²])·√(Pr[Z>tE[Z]}) by
Cauchy–Schwarz; rearrange.) At t = 0: Pr[Z>0] ≥ E[Z]²/E[Z²] = 1/(1+σ²/µ²) — a constant lower bound
on existence whenever σ² = O(µ²), even when Chebyshev's o(µ²) hypothesis fails.

## Worked threshold: subgraph appearance in G(n,p)

Let H have v vertices, e edges, density ρ(H) = e/v; H is **balanced** if every subgraph H′ has
ρ(H′) ≤ ρ(H). Let X = number of copies of H in G(n,p).

**Theorem.** If H is balanced, **p = n^{−v/e}** is the threshold for the appearance of H.

*Proof.* For each v-set S, A_S = "G|_S contains H", pᵉ ≤ Pr[A_S] ≤ v!pᵉ, so E[X] = Θ(n^v pᵉ).
- p ≪ n^{−v/e} ⇒ E[X] = o(1) ⇒ X = 0 a.a.s. (first moment).
- p ≫ n^{−v/e} ⇒ E[X] → ∞. By symmetry use Δ*; S∼T iff |S∩T| = i, 2 ≤ i ≤ v−1, with O(n^{v−i}) such
  T. Conditioning on A_S is a union over O(1) placements and Pr[A_S] = Θ(pᵉ), so a union bound over
  placement pairs only changes constants. For any fixed pair, the part of the T-copy inside S∩T is a
  subgraph of H on at most i vertices, with ≤ ie/v edges by balance; at least e − ie/v edges still have
  to be paid for, so Pr[A_T|A_S] = O(p^{e−ie/v}). Then Δ* = Σ_{i=2}^{v−1} O(n^{v−i} p^{e−ie/v}) =
  Σ_{i=2}^{v−1} O((n^v pᵉ)^{1−i/v}) = o(n^v pᵉ) = o(E[X]), since n^v pᵉ → ∞ and 1 − i/v < 1. ⇒ X > 0
  a.a.s. ∎

**Specialization (4-clique, H = K₄, v=4, e=6).** E[X] = C(n,4)p⁶ ∼ n⁴p⁶/24; threshold n^{−2/3}. Above
it, Δ* = O(n²p⁵) + O(np³): n²p⁵/(n⁴p⁶) = 1/(n²p) → 0 and np³/(n⁴p⁶) = 1/(np)³ → 0 since np → ∞. So
ω(G) ≥ 4 has threshold n^{−2/3}.

**Necessity of "balanced."** If H₁ ⊆ H has e₁/v₁ > e/v, pick v₁/e₁ < α < v/e and p = n^{−α}: then
E[#H₁] = Θ(n^{v₁−αe₁}) → 0, so a.a.s. no H₁ ⇒ no H, even though p ≫ n^{−v/e}. The threshold for general H
is n^{−v₁/e₁} with H₁ the densest subgraph (Erdős–Rényi 1960).

**Count above threshold.** For H balanced with a automorphisms and p ≫ n^{−v/e}: X ∼ n^v pᵉ / a a.a.s.

## Number-theory archetype (Turán 1934)

For x uniform in {1,…,n}, ν(x) = #distinct prime factors. With X_p = 1_{p|x}, X = Σ_{p≤M} X_p,
M = n^{1/10}: E[X] = Σ_{p≤M} 1/p + O(1) = ln ln n + O(1) (Mertens). Var[X]: diagonal = ln ln n + O(1);
for distinct p,q, Cov[X_p,X_q] = ⌊n/pq⌋/n − ⌊n/p⌋⌊n/q⌋/n² ≤ (1/n)(1/p + 1/q) (the 1/pq cancels), so
Σ_{p≠q} Cov ≤ (2M/n)Σ_{p≤M}1/p = O(n^{−9/10} ln ln n) = o(1). Writing ⌊n/r⌋/n = 1/r − θ_r/n with
0 ≤ θ_r < 1 also gives Cov[X_p,X_q] ≥ −1/n − 1/n², so the total negative contribution is
O(M²/n) = o(1). Thus Σ_{p≠q} Cov = o(1), Var[X] = ln ln n + O(1), and Chebyshev gives
Pr[|ν − ln ln n| > λ√(ln ln n)] < λ^{−2} + o(1): almost all integers have ≈ ln ln n prime factors
(Hardy–Ramanujan).

The first moment gives the disappearance half and the candidate threshold; the second moment (via
Δ* = o(E[X])) gives the appearance half and concentration; Paley–Zygmund supplies a constant lower
bound when the variance is only comparable to the mean squared. Var[X] = o(E[X]²) is the engine behind
appearance thresholds in random structures.
