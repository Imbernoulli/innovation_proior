Ramsey's theorem says every graph on N vertices has either a clique or an independent set of size about (1/2) log_2 N, and a random graph shows this bound is tight up to constants. The challenge is to exhibit such a graph by an explicit rule rather than by a coin flip. Most naive constructions control only one side: a greedy blow-up of C_5 gives polynomial-sized homogeneous sets, and Paley graphs are symmetric but we can only prove a square-root bound. The core obstacle is that bounding the clique is a statement about G, while bounding the independence number is the same statement about the complement, so a certificate of smallness must be insensitive to taking complements.

The right certificate is a dimension argument. If we make vertices combinatorial objects and encode adjacency by a condition on their intersection size, then every homogeneous family can be assigned a set of polynomials that are linearly independent because their pairwise evaluations vanish. Concretely, for a family of k-subsets of [n], assign to each set A the polynomial whose value at the characteristic vector of B is zero whenever A and B satisfy the adjacency relation. If the diagonal values are nonzero, the polynomials are independent, and their number is bounded by the dimension of the space they live in. Over the integers, however, this only bounds the clique side, because the complement condition "intersection avoids a short list" is too large: the list of allowed intersection sizes for an independent set has length comparable to k, which would make the polynomial degree too large. The breakthrough is to move to a finite field. Modulo a prime p, both "intersection ≡ -1 (mod p)" and "intersection ≢ -1 (mod p)" are bounded-size residue conditions, so the same low-degree dimension argument bounds both cliques and independent sets.

The method is the Frankl-Wilson explicit Ramsey graph. Fix a prime p and take as vertices all (p^2 - 1)-element subsets of [n]. Join two distinct subsets A and B by an edge exactly when their intersection size is congruent to -1 modulo p. The choice k = p^2 - 1 is deliberate. It is congruent to -1 modulo p, which makes the modular diagonal factor in the independent-set polynomial nonzero by Wilson's theorem. At the same time, the proper integer intersections below k that are also -1 modulo p are exactly the p - 1 values p - 1, 2p - 1, ..., p^2 - p - 1, which gives the clique side a short forbidden list over the rationals. On both sides the bound becomes the dimension of degree-(p - 1) multilinear polynomials restricted to the k-uniform layer, namely binom(n, p - 1). The constant-weight reduction is valid because Lucas' theorem guarantees the relevant binomial coefficients are nonzero modulo p.

To get concrete numbers, set n = p^3. Then the number of vertices is N = binom(p^3, p^2 - 1), whose logarithm is about p^2 log_2 p, while the clique and independence numbers are bounded by binom(p^3, p - 1), whose logarithm is about 2p log_2 p. Eliminating p gives ω(G), α(G) ≤ 2^{O(sqrt(log N · log log N))}. This is still above the probabilistic bound of 2 log_2 N, but it is sub-polynomial and the graph is fully explicit: adjacency is decided by computing one intersection size modulo p, which takes polylog(N) time. The construction therefore separates explicit Ramsey graphs from both the random-graph benchmark and the earlier polynomial-homogeneous constructions.

```python
import itertools
import math
from math import comb

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    r = int(math.isqrt(n))
    for d in range(3, r + 1, 2):
        if n % d == 0:
            return False
    return True

def frankl_wilson_graph(p: int, n: int | None = None):
    """
    Build the Frankl-Wilson explicit Ramsey graph.
    Vertices are (p**2 - 1)-subsets of [n].
    Edges join pairs whose intersection size is congruent to -1 mod p.
    Defaults to n = p**3 for the standard parameter choice.
    """
    if not is_prime(p):
        raise ValueError("p must be prime")
    if n is None:
        n = p ** 3
    k = p * p - 1
    if n < k:
        raise ValueError("n must be at least p**2 - 1")

    vertices = list(itertools.combinations(range(n), k))
    m = len(vertices)
    # adjacency stored as a list of sorted neighbor indices
    adj = [[] for _ in range(m)]

    # characteristic vectors as Python sets for fast intersection
    sets = [set(v) for v in vertices]
    target = p - 1  # -1 mod p is the same as p-1 mod p

    for i in range(m):
        si = sets[i]
        for j in range(i + 1, m):
            if (len(si & sets[j]) - target) % p == 0:
                adj[i].append(j)
                adj[j].append(i)
    return vertices, adj

def max_clique_size(adj):
    """Brute-force clique number for tiny examples."""
    n = len(adj)
    best = 0
    neigh = [set(nb) for nb in adj]
    for mask in range(1, 1 << n):
        bits = [i for i in range(n) if mask & (1 << i)]
        if len(bits) <= best:
            continue
        ok = True
        for i in range(len(bits)):
            for j in range(i + 1, len(bits)):
                if bits[j] not in neigh[bits[i]]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            best = len(bits)
    return best

def max_independent_set_size(adj):
    """Independent set = clique in the complement."""
    n = len(adj)
    comp = []
    adj_sets = [set(nb) for nb in adj]
    for i in range(n):
        comp.append([j for j in range(n) if j != i and j not in adj_sets[i]])
    return max_clique_size(comp)

# The default n = p**3 is the asymptotic choice.  For a brute-force sanity
# check we use p=2, n=5, which gives only C(5,3)=10 vertices.
if __name__ == "__main__":
    p = 2
    n = 5
    verts, adj = frankl_wilson_graph(p, n)
    k = p * p - 1
    N = len(verts)
    omega = max_clique_size(adj)
    alpha = max_independent_set_size(adj)
    theory_bound = comb(n, p - 1)
    print(f"p={p}, n={n}, k={k}, vertices={N}")
    print(f"clique number {omega}, independence number {alpha}")
    print(f"theory bound binom(n, p-1) = {theory_bound}")
    assert omega <= theory_bound and alpha <= theory_bound
```
