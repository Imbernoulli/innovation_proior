# Context: Explicit constructions of graphs with small clique and independence numbers

## Research question

Ramsey's theorem guarantees that every 2-coloring of the edges of a large complete graph contains a large monochromatic clique. In the graph language: for every graph G on N vertices, either the clique number ω(G) or the independence number α(G) is at least about (1/2)·log₂ N. A graph is "Ramsey" — a good lower-bound witness for the Ramsey number R(t) — when *both* ω(G) and α(G) are small, ideally as small as the theorem allows.

The precise task: **construct, by an explicit deterministic rule (no randomness), a graph on N vertices in which both the largest clique and the largest independent set have size only polylogarithmic, or at worst quasi-polynomial, in N.** "Explicit" means the construction is described by a formula or short algorithm — ideally the adjacency of two vertices is decidable in time polylog(N) — rather than produced by a random experiment or an exhaustive search. The yardstick is the probabilistic existence bound (graphs with ω, α ≤ 2·log₂ N + O(1) exist); the goal is to approach it by a construction one can write down.

Why it matters: it is a central problem in explicit combinatorial constructions, posed by Erdős, who offered a prize for it. Existence is easy; *exhibiting* such a graph is the hard part, and the gap between what randomness proves to exist and what we can build by hand measures how far our explicit-construction toolkit lags behind pure existence.

## Background

**The two-sided difficulty.** For a single fixed graph, controlling ω alone is usually easy and controlling α alone is usually easy, but controlling *both at once* is what makes the problem resist every naive attempt. Bounding ω is a statement about G; bounding α is the same statement about the complement Ḡ. A construction succeeds only if its certificate is symmetric enough to bind G and Ḡ simultaneously.

**The upper bound (Ramsey/Erdős–Szekeres).** A greedy "all-or-nothing sequence" argument shows every graph on 2^{2t} vertices has ω ≥ t or α ≥ t, so R(t) ≤ 2^{2t}; equivalently every N-vertex graph has max(ω, α) ≥ (1/2)·log₂ N. This is the obstacle a good construction must push against: you can never get both below ~log₂ N.

**The probabilistic existence bound (Erdős 1947).** Take N vertices and include each edge independently with probability 1/2. A fixed set of t vertices is a clique with probability 2^{-binom(t,2)}; there are at most binom(N,t) ≤ N^t such sets, so the expected number of t-cliques is at most N^t · 2^{-t(t-1)/2}. The same bound holds for independent sets. Once t is slightly larger than 2·log₂ N, the union-bound expectation is below 1, so with positive probability the random graph has neither a t-clique nor a t-independent-set. Hence graphs with ω, α ≤ 2·log₂ N + O(1) **exist**. The argument is non-constructive: it gives no rule for the edges, and certifying a candidate by checking all t-subsets costs N^{Θ(t)}, exponential in log²N when t is logarithmic.

**The linear-algebra (dimension) method in extremal set theory.** A family of objects can be bounded in size by assigning to each object a vector or polynomial in a fixed space and proving the assigned objects are linearly independent; the family size is then at most the dimension of the space. The space of *multilinear* polynomials of degree at most d in n variables has dimension Σ_{i=0}^{d} binom(n,i), because a multilinear monomial is indexed by the subset of variables it contains, and only subsets of size ≤ d are allowed. On a constant-weight layer over a characteristic-zero field, this can sharpen to binom(n,d): if every evaluation point has |x| = k, then for |I| = r < d,
  Σ_{J⊇I, |J|=d} x_J = binom(k-r,d-r) x_I,
so whenever the scalar is nonzero in the coefficient field, lower-degree monomials are redundant as functions on that layer. Over a finite field F_p the nonzero-scalar check becomes part of the theorem, and Lucas' theorem is the usual way to verify it for the chosen weights. Modular arithmetic of intersection sizes then designs the vanishing polynomials. This method is the engine behind restricted-intersection theorems for set systems.

**Restricted-intersection theorems for set systems.** Consider a family F of k-element subsets of [n].
- *Ray–Chaudhuri–Wilson (non-modular).* If there is a set L of s integers such that every pairwise intersection |A∩B| (A≠B in F) lies in L, then |F| ≤ binom(n,s). One assigns to each A the polynomial ∏_{ℓ∈L}(⟨x,v_A⟩ − ℓ) of degree s in the indeterminate vector x and works over a characteristic-zero field; evaluated at characteristic vectors these are diagonal-nonzero and off-diagonal-zero. The polynomials are independent, and on the k-uniform layer their restrictions lie in the span of the degree-s monomials, whose dimension is binom(n,s).
- *Frankl–Wilson modular version.* If p is prime, the common set size k and the allowed intersection residues μ₁,…,μ_s (mod p) satisfy μ_i ≢ k (mod p) for all i, every pairwise intersection is congruent to one of the μ_i, and binom(k-r,s-r) is nonzero in F_p for 0 ≤ r < s, then the same constant-weight dimension proof gives |F| ≤ binom(n,s). The polynomial is evaluated over F_p: the off-diagonal factors vanish because the intersection hits an allowed residue, while the diagonal factor ∏(k − μ_i) is nonzero mod p precisely because k avoids every μ_i.

The crucial feature is that a *single* modulus p can make a complement condition into a bounded-size residue list. If an edge rule pins intersections to one residue class, then an independent set has intersections in the other p−1 residues, a modular restricted-intersection condition whose length no longer grows with k. The clique side can be handled in characteristic zero using the actual integer intersection values: proper intersections in one residue class form a short list below the common set size k. The remaining design problem is to choose k so the modular diagonal is nonzero and the integer list is short enough to give the same dimension scale on both sides.

## Baselines

**Trivial / greedy explicit graphs.** Iterated blow-ups of the 5-cycle (replace each vertex of C₅ by a copy of C₅) give an explicit family with ω = α = 2^t on 5^t vertices, i.e. ω, α ≈ N^{0.43}. Far from logarithmic — these only prove weak Ramsey lower bounds like R(t) ≥ a small power. Representative of "constructions by hand" before algebraic methods: polynomially many vertices buy only a polynomial-sized homogeneous set.

**Paley graphs.** For a prime q ≡ 1 (mod 4), join x,y ∈ F_q when x−y is a quadratic residue. These are self-complementary (so ω = α by symmetry) and *conjectured* to have ω, α = O(polylog q). But the best unconditional bound is only ω, α = O(√q): the algebra (character-sum / Weil bounds) does not deliver a polylog certificate. A candidate, not a proof — illustrating that an explicit, symmetric, algebraically defined graph still need not come with a provable two-sided bound.

**The probabilistic graph itself.** The benchmark to be matched: ω, α ≤ 2·log₂ N + O(1), but only as an existence statement. Any explicit construction is measured by how close its provable bound comes to this.

## Evaluation settings

The object is a graph G on N vertices. The metrics are ω(G) and α(G) (equivalently the largest monochromatic clique over the 2-coloring G/Ḡ), reported as functions of N, and the explicitness of the adjacency rule (computable in polylog(N), in poly(N), or only by search). The natural comparison points are the trivial blow-up family, the Paley-graph candidate, and the probabilistic existence bound 2·log₂ N + O(1). Correctness for a given construction is checkable on small instances by enumerating all subsets that could form a clique or independent set and confirming none exceeds the claimed bound; the asymptotic claim is then a theorem about the family. There is no training/test split — the deliverable is a graph and a proof.

## Code framework

The construction is fixed in its scaffolding and open only at the two places where the design choice lives. The pieces that are pinned down in advance:

    Ground set (vertices):
        choose a prime p and uniform parameters n and k, and let the vertex
        set be a family of k-element subsets of an n-element ground set
        (the constant-weight layer). N = |family| is the vertex count; the
        target is a homogeneous-set bound that is polylog(N).

    Adjacency rule (open slot 1):
        a symmetric predicate on a pair of distinct vertices (A, B) that
        depends only on the intersection size |A ∩ B|, sorted by its residue
        modulo the prime p. One residue class of |A ∩ B| is declared "edge,"
        the rest "non-edge." WHICH residue (and how k itself sits modulo p)
        is the empty slot — it must be chosen so that the modular diagonal
        ∏(k − μ_i) stays nonzero in F_p and the integer intersection list on
        the complementary side stays short.

    Bound to be proved (open slot 2):
        the clique number ω and the independence number α. A clique is a
        sub-family whose pairwise intersections all land in the "edge"
        residue class; an independent set is one whose pairwise intersections
        avoid it. Each side is to be bounded by a restricted-intersection /
        constant-weight dimension argument: assign to each set a vanishing
        polynomial, show the assigned polynomials are linearly independent,
        and read off |sub-family| ≤ dimension of the space of low-degree
        multilinear monomials on the layer. The two bounds, expressed back
        in N, are the quantities the design of slot 1 must make both small.

    Verification (small instances):
        enumerate the family for small (n, k, p), build the adjacency table
        from the chosen predicate, exhaustively search for the largest clique
        and largest independent set, and confirm neither exceeds the bound
        predicted by the dimension count — a sanity check on the algebra, not
        the asymptotic proof.

The linear-algebra (polynomial dimension) method is the fixed tool; the only freedom is the intersection predicate of slot 1 and the matching pair of homogeneous-set bounds of slot 2. The mod-p rule that simultaneously controls both ω and α is exactly what the method must supply, and is therefore left open here.
