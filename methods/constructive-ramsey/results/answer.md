# The Frankl–Wilson explicit Ramsey graph

## Problem

Construct, by an explicit deterministic rule, a graph G on N vertices in which both the clique number ω(G) and the independence number α(G) are sub-polynomial in N — far below the polynomial-sized homogeneous sets that hand constructions give, though still above the probabilistic existence bound ω, α ≤ 2·log₂ N + O(1). The adjacency of two vertices must be decidable in time polylog(N); no randomness, no search.

## Key idea

Make the vertices k-element subsets of a ground set [n] and define adjacency through the intersection size modulo a prime p. Bound the independent-set side by the **Frankl–Wilson modular intersection theorem** and the clique side by the same polynomial dimension argument over characteristic zero: assign to each set a low-degree polynomial in the characteristic vectors, show the polynomials attached to a homogeneous family are linearly independent over a field, and conclude the family is no larger than the dimension of the polynomial space.

The decisive point is that working **modulo a prime** makes the complement-side condition a bounded-size residue list, while the clique-side condition becomes a short integer list:
- A clique requires every pairwise |A∩B| ≡ −1 (mod p), and since proper intersections are below k = p²−1, the possible integer values are {p−1,2p−1,…,p²−p−1}.
- An independent set requires every pairwise |A∩B| ≢ −1 (mod p), i.e. the p−1 residues {0,…,p−2}.

Both are restricted-intersection families bounded by the same binomial dimension. Choosing the set size k = p²−1 makes the construction work on both sides at once: k ≡ −1 (mod p) keeps the modular diagonal factor nonzero (via Wilson's theorem (p−1)! ≡ −1), Lucas' theorem keeps the constant-weight reduction nondegenerate in F_p, and the actual integer intersection sizes ≡ −1 (mod p) below k are precisely p−1 values.

## The construction

Fix a prime p and a ground set [n].
- **Vertices:** all (p²−1)-element subsets of [n]. Count N = binom(n, p²−1).
- **Edges:** join A and B iff |A∩B| ≡ −1 (mod p).

## The bound

For each set A let v_A ∈ {0,1}^n be its characteristic vector, so ⟨v_A, v_B⟩ = |A∩B|.

*Independent set (p−1 vanishing residues, over F_p).* For A in an independent set define
  Q_A(x) = ∏_{μ=0}^{p−2} (⟨x, v_A⟩ − μ)  over F_p, degree p−1.
For B ≠ A, |A∩B| lands in one of {0,…,p−2}, so Q_A(v_B) = 0. On the diagonal ⟨v_A,v_A⟩ = p²−1 ≡ −1, so Q_A(v_A) = ∏_{μ=0}^{p−2}(−1−μ) = (−1)^{p−1}(p−1)! ≠ 0 (Wilson). If Σ c_A Q_A vanished on the family, evaluating at v_B leaves c_B Q_B(v_B), so every c_B is zero; the Q_A are linearly independent. Their restrictions to the k-subset layer lie in the span of the degree-(p−1) monomials: for |I| = r < p−1,
  Σ_{J⊇I, |J|=p−1} x_J = binom(p²−1−r,p−1−r)x_I,
and Lucas' theorem gives binom(p²−1−r,p−1−r) ≡ binom(p−1,0)binom(p−1−r,p−1−r) ≡ 1 (mod p). Hence α(G) ≤ binom(n,p−1).

*Clique (p−1 integer intersection values, over characteristic zero).* A clique is an L-intersecting family with L = {p−1, 2p−1, …, p²−p−1}, |L| = p−1, and set size k = p²−1 ∉ L. For A in the clique define P_A(x) = ∏_{ℓ∈L}(⟨x,v_A⟩ − ℓ). Then P_A(v_B)=0 for B≠A and P_A(v_A)=∏_{ℓ∈L}(k−ℓ)≠0, so the same diagonal evaluation proves linear independence. On the k-subset layer, the lower-degree monomial relation has nonzero coefficient binom(k-r,p−1-r), so the restrictions lie in the span of the degree-(p−1) monomials. Hence ω(G) ≤ binom(n,p−1).

So both are ≤ binom(n, p−1), i.e. O_p(N^{1/(p+1)}).

## The resulting Ramsey bound

Take n = p³. Then N = binom(p³, p²−1) with log₂ N = (1+o(1))p² log₂ p, while the homogeneous-set bound binom(p³, p−1) has log₂ = (2+o(1))p log₂ p. Eliminating p gives

  ω(G), α(G) ≤ 2^{O(√(log N · log log N))}.

Equivalently, the Ramsey number satisfies R(t) ≥ t^{Ω(log t / log log t)}. The adjacency rule costs polylog(N) per pair — very explicit. This is exp(√(log N log log N)) above the probabilistic existence bound 2·log₂ N + O(1), but far below polynomial-sized homogeneous sets.


