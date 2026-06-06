OK, let me think this through from scratch. I want a graph on N vertices where both the largest clique and the largest independent set are tiny — polylog in N if I'm lucky, quasi-polynomial if I'm honest. And I have to *write it down*: a rule, a formula, not a coin.

The annoying thing is that I already know such graphs exist. Flip a fair coin for every edge of K_N. A fixed t-set is all-edges with probability 2^{-t(t-1)/2}; there are at most N^t of them, so the expected number of t-cliques is at most N^t · 2^{-t(t-1)/2}, and the same for independent sets. Set t ≈ 2 log₂ N and that product is way below 1, so a graph with ω, α ≤ 2 log₂ N exists. Beautiful, and completely useless to me as a builder — it names no edges. To make it explicit I'd have to *search*: certify the property over all N^{Θ(t)} candidate homogeneous sets, which is exponential in the number of vertices. That's the brute-force output I refuse to call a construction. I want the graph to fall out of structure, not out of a search.

So let me feel the actual obstacle. Bounding ω is a statement about my graph G. Bounding α is the *same* statement about the complement Ḡ. The random graph nails both because flipping coins is symmetric under complementation — but that symmetry is exactly what I can't certify by hand. If I cook up some clever graph and manage to argue "no large clique," I've spent my cleverness on G; nothing says the *same* argument transfers to Ḡ. Most explicit attempts die right here: they control one side and lose the other. Paley graphs, say — quadratic residues over F_q — are self-complementary, so ω = α for free, and people believe they're polylog. But the only bound anyone can prove is O(√q) from character sums. The algebra is symmetric but it isn't *strong*: it gives no logarithmic certificate. And iterated blow-ups of C₅ are honestly explicit but give ω = α ≈ N^{0.43} — polynomially large, nowhere near log. So "symmetric" isn't enough and "explicit" isn't enough; I need a *certificate of smallness* that is insensitive to which side I'm bounding.

Let me ask what kind of certificate could even bound the number of vertices in a structured family. If I could attach to each vertex a vector, and prove that the vectors attached to any clique are linearly independent, then the clique can't be bigger than the dimension of the space those vectors live in. That's a real handle: cliques bounded by a *dimension*. And dimensions of nice spaces are small and computable. So the plan crystallizes into: make the vertices combinatorial objects, and make "being adjacent" a condition that forces independence of some assigned vectors.

What objects? Subsets feel right, because for sets the natural pairing — the size of the intersection |A∩B| = ⟨v_A, v_B⟩ of characteristic vectors — is an inner product, and inner products are what polynomials in those vectors can read off. So: vertices are k-subsets of [n], and adjacency is some condition on |A∩B|.

Suppose every pair in a clique had its intersection size confined to a small set L of values. Assign to each set A the polynomial

  P_A(x) = ∏_{ℓ∈L} (⟨x, v_A⟩ − ℓ),

a polynomial of degree |L| in the formal vector x. Evaluate it at the characteristic vector v_B of another set B in the clique. Then ⟨v_B, v_A⟩ = |A∩B| ∈ L, so one factor is zero — P_A(v_B) = 0 for B ≠ A. But P_A(v_A) = ∏(k − ℓ); if k ∉ L this is a product of nonzero numbers, nonzero. So the matrix [P_A(v_B)] is diagonal-nonzero, off-diagonal-zero — the P_A are linearly independent. They have degree |L|. After reducing x_i² to x_i, because I only plug in 0/1 vectors, they are multilinear of degree at most |L|. At first that gives the full degree-≤|L| dimension Σ_{i=0}^{|L|} binom(n,i), but I can do better because all evaluation points have the same weight k. If a monomial is x_I with |I| = r < s := |L|, then on the k-subset layer

  Σ_{J⊇I, |J|=s} x_J = binom(k-r, s-r) x_I.

Over the integers that scalar is nonzero, so every lower-degree monomial is a linear combination of degree-s monomials as a function on the layer. The relevant dimension is therefore just the number of degree-s monomials, binom(n,s). So the clique has at most binom(n,|L|) vertices.

That's the Ray–Chaudhuri–Wilson bound and it's genuinely a *certificate*: an L-intersecting family of k-sets, with k ∉ L, has size ≤ binom(n,|L|). If I could force a clique to be L-intersecting with |L| small, ω would be small.

But wait — this only handled the clique. The independent set is a family with *no* edges, and if I defined adjacency as "intersection in L," then an independent set is a family whose intersections all *avoid* L. Avoiding L is not the same shape as lying in a small set; the complement of L inside {0,1,…,k−1} could be huge. So the off-diagonal vanishing trick doesn't fire for the independent set, and I'm back to controlling one side only. The dimension method is strong but it looks one-sided. This is the wall.

Let me stare at why it's one-sided. The trick needs, for every off-diagonal pair, *a single algebraic reason the product vanishes* — one factor hitting zero. For the clique that reason was "the intersection is one of finitely many forbidden values." For the independent set I'd need the *negation* — "the intersection is NOT one of those values" — to also be expressible as finitely many forbidden values that a degree-small polynomial can annihilate. Over the integers that negation is an infinite set. Over the integers I'm stuck.

So work modulo a prime. Intersection sizes are integers, but I only ever care about them through whether a polynomial vanishes, and vanishing is a statement over a field. Over F_p there are only p residues. Now "≡ −1 (mod p)" and "≢ −1 (mod p)" are *both* finite conditions: the first is a single residue, the second is the p−1 residues {0,1,…,p−2}. Each is a small forbidden set mod p, and each is exactly the kind of thing ∏(⟨x,v_A⟩ − μ) can annihilate with a low-degree polynomial. The negation stopped being infinite the moment I quotiented by p. *That* is the symmetry I needed — not in the graph, in the arithmetic.

Let me pick the adjacency rule to exploit this. Say A and B are adjacent iff |A∩B| ≡ −1 (mod p). Then:

A *clique* is a family where every pairwise |A∩B| ≡ −1 (mod p). The intersection residue is pinned to a single value, −1.

An *independent set* is a family where every pairwise |A∩B| ≢ −1 (mod p), i.e. the residue lands in {0,1,…,p−2}. That's p−1 forbidden residues for the diagonal to dodge, a degree-(p−1) modular condition on each pair.

Both are restricted-intersection families mod p. Now I need the diagonal to survive in both cases — I need k = |A| to *not* collide with the forbidden residues. For the clique, the forbidden value is −1, so I'd better have k ≢ −1… no wait, let me redo this carefully, because the diagonal/off-diagonal logic flips depending on which family I'm bounding, and getting the sign wrong here would wreck everything.

Take the independent-set side first since it's the heavier one. The family has all pairwise intersections ≡ μ for some μ ∈ {0,…,p−2}. To run the polynomial bound I assign

  Q_A(x) = ∏_{μ=0}^{p−2} (⟨x, v_A⟩ − μ)   over F_p,

degree p−1. For B ≠ A in the independent set, ⟨v_B,v_A⟩ = |A∩B| ≡ some μ in {0,…,p−2}, so a factor vanishes: Q_A(v_B) = 0. For the diagonal, ⟨v_A,v_A⟩ = |A| = k, and I need k ≢ μ for *every* μ ∈ {0,…,p−2}, i.e. k ≡ −1 (mod p), the *one* residue I left out. Then

  Q_A(v_A) = ∏_{μ=0}^{p−2}(−1 − μ) = ∏_{μ=0}^{p−2} −(1+μ) = (−1)^{p−1} (p−1)! ,

and by Wilson's theorem (p−1)! ≡ −1 (mod p), nonzero. So the diagonal survives precisely because I chose the set size k ≡ −1 (mod p) — the residue the independent-set family forbids. The Q_A are independent over F_p. I still need the sharp dimension, not the looser Σ_{i=0}^{p−1} binom(n,i), and this is where the constant-weight layer matters again. For |I| = r < p−1,

  Σ_{J⊇I, |J|=p−1} x_J = binom(k-r, p−1-r) x_I

on every k-subset. Here k = p²−1, so Lucas' theorem gives

  binom(p²−1−r, p−1−r) ≡ binom(p−1,0) binom(p−1−r,p−1−r) ≡ 1  (mod p).

The scalar is nonzero in F_p, so the restrictions of the Q_A to the k-subset layer lie in the span of the degree-(p−1) monomials. That space has dimension binom(n,p−1). Therefore α ≤ binom(n,p−1).

Now the clique side, and here I have to be careful about integers vs residues. The clique forbids… no, the clique *requires* every intersection ≡ −1 (mod p). The set size is also ≡ −1 (mod p), so naively the diagonal residue equals the off-diagonal residue and the modular trick can't separate them — I can't put −1 in the "forbidden" list and also have k avoid it, because k ≡ −1 too. So the *modular* bound doesn't directly bound the clique. Fine — for the clique I don't need modular cleverness, because the clique is L-intersecting over the *integers*. Every |A∩B| is ≡ −1 (mod p) and is a proper-intersection value strictly less than k = p²−1, so as an integer it's one of −1+p, −1+2p, …, i.e. {p−1, 2p−1, …, p²−p−1}. That's exactly p−1 distinct integer values. So L = {p−1, 2p−1, …, p²−p−1}, |L| = p−1, and the integer set size k = p²−1 is bigger than every element of L, hence k ∉ L. Ray–Chaudhuri–Wilson over the integers gives clique ≤ binom(n, p−1). Same bound, by the non-modular twin of the same dimension argument.

So I need k ≡ −1 (mod p) for the independent-set side, and I need k large enough that the integer intersection values ≡ −1 (mod p) below k number exactly p−1 for the clique side — which happens when k = p²−1: the values are (p−1), (2p−1), …, ((p−1)p − 1) = p²−p−1, that's p−1 of them. And p²−1 = (p−1)(p+1) ≡ −1 (mod p). So **k = p²−1** does both jobs at once. That's why the set size is p²−1 and not just "anything ≡ −1": it's the smallest natural size that makes the clique's integer-intersection set have exactly p−1 values while keeping the modular diagonal alive. Both sides land on the *same* number, p−1, and that's the whole miracle — the modulus p hands me p−1 constraint values whether I'm looking at the clique or its complement.

Let me write the construction down cleanly. Fix a prime p. Ground set [n]. Vertices V = all (p²−1)-element subsets of [n]. Join A,B iff |A∩B| ≡ −1 (mod p). The number of vertices is

  N = binom(n, p²−1),

and I've just shown

  ω(G) ≤ binom(n, p−1)   and   α(G) ≤ binom(n, p−1).

Both clique and independent set are bounded by binom(n,p−1). Now what does that buy me in terms of N? I get to choose n; the cleanest choice is n = p³, which makes the bookkeeping come out in a recognizable form. Let me push the numbers.

With n = p³, the homogeneous-set bound is binom(p³, p−1). Since p−1 is tiny compared with p³,

  log₂ binom(p³,p−1) = (p−1)·log₂(p³) − log₂((p−1)!) + O(1)
                         = (2+o(1))p log₂ p.

And the vertex count:

  log₂ N = log₂ binom(p³, p²−1)
         = (p²−1)·log₂(p³) − log₂((p²−1)!) + O(p²)
         = (1+o(1))p² log₂ p,

since binom(p³, p²−1) ≈ (p³/(p²))^{p²} up to lower order, i.e. the dominant exponent is p²·log p. So if I write L := log₂ N = Θ(p² log p), then p ≈ √(L / log L) up to constants, and the homogeneous-set bound is

  log₂ max(ω,α) ≤ Θ(p log p) = Θ( √(L/log L) · log L ) = Θ( √(L · log L) ) = Θ( √(log N · log log N) ).

So ω(G), α(G) ≤ 2^{O(√(log N · log log N))}. Compare to the existence bound 2 log₂ N — I'm at exp(√(log N · log log N)) instead of polylog, so I haven't reached the random graph, but I have pushed far below polynomial-sized homogeneous sets: the homogeneous sets are sub-polynomial, quasi-polynomially small in N. As a Ramsey lower bound this reads R(t) ≥ t^{Ω(log t / log log t)}, super-polynomial in t.

And it's *very* explicit: to decide whether two vertices are adjacent I list two (p²−1)-subsets and compute their intersection size mod p — polylog(N) work, no search anywhere. The entire content of "no large clique and no large independent set" was discharged by one dimension count, applied once over the integers and once over F_p, with the prime p doing the work of making the complement-side condition finite.

Let me double-check the load-bearing step, the Wilson-theorem diagonal, because if that factor were zero the whole independent-set bound collapses. Q_A(v_A) = ∏_{μ=0}^{p−2}(k − μ) with k ≡ −1 (mod p). Substituting k ≡ −1: ∏_{μ=0}^{p−2}(−1 − μ). For μ = 0 this is −1; μ=1 gives −2; …; μ=p−2 gives −(p−1). So the product is (−1)·(−2)·…·(−(p−1)) = (−1)^{p−1}·(p−1)!. Wilson's theorem says (p−1)! ≡ −1 (mod p), and the product is nonzero for every prime p, including p = 2. Good — the diagonal is genuinely nonzero, the independence of the Q_A holds, and the bound stands. (For the clique side over the integers there's no modular subtlety: ∏_{ℓ∈L}(k − ℓ) with k = p²−1 larger than every ℓ ∈ L is a product of positive integers, nonzero.)

Now the code that actually builds and certifies this. Vertices are the (p²−1)-subsets; adjacency is intersection ≡ p−1 (mod p); and on small instances I brute-force ω and α to confirm they sit under binom(n,p−1). For p = 2 the set size is p²−1 = 3 and adjacency is "intersection ≡ 1 (mod 2)", i.e. odd intersection — small enough to enumerate and watch the bound hold.

```python
from itertools import combinations
from math import comb

def vertices(n, k):
    return [frozenset(s) for s in combinations(range(n), k)]

def adjacent(A, B, rule_parameter):
    p = rule_parameter
    return (len(A & B) % p) == (p - 1)              # |A cap B| = -1 (mod p)

def build_graph(n, k, rule_parameter):
    verts = vertices(n, k)
    N = len(verts)
    adj = [[False] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            adj[i][j] = adj[j][i] = adjacent(verts[i], verts[j], rule_parameter)
    return verts, adj

def max_homogeneous(adj, want_edge):
    # largest clique (want_edge=True) / independent set (want_edge=False)
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
    p = rule_parameter
    if k != p * p - 1:
        raise ValueError("this bound uses k = p^2 - 1")
    return comb(n, p - 1)

if __name__ == "__main__":
    for p, n in [(2, 6), (2, 7), (2, 8)]:
        k = p * p - 1                               # k = -1 (mod p)
        verts, adj = build_graph(n, k, p)
        N = len(verts)
        bound = claimed_bound(n, k, p)
        w = max_homogeneous(adj, True)               # clique  <= bound
        a = max_homogeneous(adj, False)              # indep.  <= bound
        print(f"p={p} n={n}: N=binom({n},{p*p-1})={N}  "
              f"omega={w} alpha={a}  bound=binom(n,p-1)={bound}  ok={w<=bound and a<=bound}")
```

The causal chain, end to end: I wanted both sides of the graph small, but the dimension method that bounds a clique is one-sided over the integers because the *complement* condition on intersections is infinite there; passing to a prime modulus p makes "intersection ≢ −1 (mod p)" a finite condition (p−1 residues) just like "intersection ≡ −1 (mod p)", so the *same* low-degree-polynomial dimension argument bounds clique and independent set alike; choosing the set size p²−1 simultaneously keeps the modular diagonal alive (size ≡ −1 mod p) and pins the clique's integer-intersection set to p−1 values, so both bounds collapse to binom(n,p−1); and with n = p³ that turns into ω, α ≤ 2^{O(√(log N · log log N))} on N = binom(p³, p²−1) vertices, decidable per edge in polylog(N) — an explicit Ramsey graph with sub-polynomial homogeneous sets, still short of the probabilistic existence bound by an exp(√) factor.
