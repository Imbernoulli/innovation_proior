# Context: Explicit constructions of graphs with small clique and independence numbers

## Research question

Ramsey's theorem guarantees that every 2-coloring of the edges of a large complete graph contains a large monochromatic clique. In the graph language: for every graph G on N vertices, either the clique number ω(G) or the independence number α(G) is at least about (1/2)·log₂ N. A graph is "Ramsey" — a good lower-bound witness for the Ramsey number R(t) — when *both* ω(G) and α(G) are small, ideally as small as the theorem allows.

The precise task: **construct, by an explicit deterministic rule (no randomness), a graph on N vertices in which both the largest clique and the largest independent set have size only polylogarithmic, or at worst quasi-polynomial, in N.** "Explicit" means the construction is described by a formula or short algorithm — ideally the adjacency of two vertices is decidable in time polylog(N) — rather than produced by a random experiment or an exhaustive search. The yardstick is the probabilistic existence bound (graphs with ω, α ≤ 2·log₂ N exist); the goal is to approach it by a construction one can write down.

Why it matters: it is a central problem in explicit combinatorial constructions, posed by Erdős, who offered a prize for it. Existence is easy; *exhibiting* such a graph is the hard part, and the gap between what randomness proves to exist and what we can build by hand measures how far our explicit-construction toolkit lags behind pure existence.

## Background

**The two-sided difficulty.** For a single fixed graph, controlling ω alone is usually easy and controlling α alone is usually easy, but controlling *both at once* is what makes the problem resist every naive attempt. Bounding ω is a statement about G; bounding α is the same statement about the complement Ḡ. A construction succeeds only if its certificate is symmetric enough to bind G and Ḡ simultaneously.

**The upper bound (Ramsey/Erdős–Szekeres).** A greedy "all-or-nothing sequence" argument shows every graph on 2^{2t} vertices has ω ≥ t or α ≥ t, so R(t) ≤ 2^{2t}; equivalently every N-vertex graph has max(ω, α) ≥ (1/2)·log₂ N. This is the obstacle a good construction must push against: you can never get both below ~log₂ N.

**The probabilistic existence bound (Erdős 1947).** Take N vertices and include each edge independently with probability 1/2. A fixed set of t vertices is a clique with probability 2^{-binom(t,2)}; there are at most binom(N,t) ≤ N^t such sets, so the expected number of t-cliques is at most N^t · 2^{-t(t-1)/2}. The same bound holds for independent sets. When t ≈ 2·log₂ N this product is far below 1, so with positive probability the random graph has neither a t-clique nor a t-independent-set. Hence graphs with ω, α ≤ 2·log₂ N **exist**. The argument is non-constructive: it gives no rule for the edges, and derandomizing it — searching N^{2} candidate edges to certify the property — costs time exponential in the number of vertices.

**The linear-algebra (dimension) method in extremal set theory.** A family of objects can be bounded in size by assigning to each object a vector or polynomial in a fixed space and proving the assigned objects are linearly independent; the family size is then at most the dimension of the space. The space of *multilinear* polynomials of degree at most d in n variables has dimension Σ_{i=0}^{d} binom(n,i), because a multilinear monomial is indexed by the subset of variables it contains, and only subsets of size ≤ d are allowed. On a constant-weight layer, this can sharpen to binom(n,d): if every evaluation point has |x| = k, then for |I| = r < d,
  Σ_{J⊇I, |J|=d} x_J = binom(k-r,d-r) x_I,
so whenever the scalar is nonzero, lower-degree monomials are redundant as functions on that layer. Over a finite field F_p one can use modular arithmetic of intersection sizes to design the polynomials. This method is the engine behind restricted-intersection theorems for set systems.

**Restricted-intersection theorems for set systems.** Consider a family F of k-element subsets of [n].
- *Ray–Chaudhuri–Wilson (non-modular).* If there is a set L of s integers such that every pairwise intersection |A∩B| (A≠B in F) lies in L, then |F| ≤ binom(n,s). One assigns to each A the polynomial ∏_{ℓ∈L}(⟨x,v_A⟩ − ℓ) of degree s in the indeterminate vector x; evaluated at characteristic vectors these are diagonal-nonzero and off-diagonal-zero. The polynomials are independent, and on the k-uniform layer their restrictions lie in the span of the degree-s monomials, whose dimension is binom(n,s).
- *Modular version.* If p is prime, the common set size k and the allowed intersection residues μ₁,…,μ_s (mod p) satisfy μ_i ≢ k (mod p) for all i, and every pairwise intersection is ≡ some μ_i (mod p), then the same constant-weight dimension proof gives |F| ≤ binom(n,s) when the layer-homogenization scalars are nonzero. The polynomial is evaluated over F_p: the off-diagonal factors vanish because the intersection hits an allowed residue, while the diagonal factor ∏(k − μ_i) is nonzero mod p precisely because k avoids every μ_i.

The crucial feature is that a *single* modulus p can make a complement condition finite. If an edge rule pins intersections to one residue class, then an independent set has intersections in the other p−1 residues, a modular restricted-intersection condition. The clique side can be handled over the integers: proper intersections in one residue class form a finite list below the common set size k. The remaining design problem is to choose k so the modular diagonal is nonzero and the integer list is short enough to give the same dimension scale on both sides.

## Baselines

**Trivial / greedy explicit graphs.** Iterated blow-ups of the 5-cycle (replace each vertex of C₅ by a copy of C₅) give an explicit family with ω = α = 2^t on 5^t vertices, i.e. ω, α ≈ N^{0.43}. Far from logarithmic — these only prove weak Ramsey lower bounds like R(t) ≥ a small power. Representative of "constructions by hand" before algebraic methods: polynomially many vertices buy only a polynomial-sized homogeneous set.

**Paley graphs.** For a prime q ≡ 1 (mod 4), join x,y ∈ F_q when x−y is a quadratic residue. These are self-complementary (so ω = α by symmetry) and *conjectured* to have ω, α = O(polylog q). But the best unconditional bound is only ω, α = O(√q): the algebra (character-sum / Weil bounds) does not deliver a polylog certificate. A candidate, not a proof — illustrating that an explicit, symmetric, algebraically defined graph still need not come with a provable two-sided bound.

**The probabilistic graph itself.** The benchmark to be matched: ω, α ≤ 2·log₂ N, but only as an existence statement. Any explicit construction is measured by how close its provable bound comes to this.

## Evaluation settings

The object is a graph G on N vertices. The metrics are ω(G) and α(G) (equivalently the largest monochromatic clique over the 2-coloring G/Ḡ), reported as functions of N, and the explicitness of the adjacency rule (computable in polylog(N), in poly(N), or only by search). The natural comparison points are the trivial blow-up family, the Paley-graph candidate, and the probabilistic existence bound 2·log₂ N. Correctness for a given construction is checkable on small instances by enumerating all subsets that could form a clique or independent set and confirming none exceeds the claimed bound; the asymptotic claim is then a theorem about the family. There is no training/test split — the deliverable is a graph and a proof.

## Code framework

A minimal verification harness needs a finite vertex list, a pairwise predicate, an adjacency table, and a brute-force homogeneous-set finder for small instances.

```python
from itertools import combinations
from math import comb

def vertices(n, k):
    # k-element subsets of [n], represented as frozensets.
    return [frozenset(s) for s in combinations(range(n), k)]

def adjacent(A, B, rule_parameter):
    # TODO: explicit predicate on the pair (A, B).
    raise NotImplementedError

def build_graph(n, k, rule_parameter):
    verts = vertices(n, k)
    N = len(verts)
    adj = [[False] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            e = adjacent(verts[i], verts[j], rule_parameter)
            adj[i][j] = adj[j][i] = e
    return verts, adj

def max_homogeneous(adj, want_edge):
    # Brute-force largest clique (want_edge=True) / independent set
    # (want_edge=False), for verifying the bound on small instances.
    N = len(adj); best = 0
    def extend(chosen, cand):
        nonlocal best
        best = max(best, len(chosen))
        for x in list(cand):
            extend(chosen + [x],
                   [y for y in cand if y > x and adj[x][y] == want_edge])
    extend([], list(range(N)))
    return best

def claimed_bound(n, k, rule_parameter):
    # TODO: homogeneous-set bound proved from the same parameters.
    raise NotImplementedError
```
