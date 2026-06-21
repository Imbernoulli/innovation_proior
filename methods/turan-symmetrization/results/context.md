# Context

## Research question

Fix a number of vertices $n$ and a forbidden complete subgraph. The question is purely extremal: among all simple graphs on $n$ vertices that contain **no clique on $r+1$ vertices** (no $K_{r+1}$ as a subgraph), how many edges can a graph have, and *which* graphs achieve the maximum?

This matters because it is the prototype of a whole genre of question — "forbid a fixed subgraph $H$; how many edges force a copy of $H$?" — and because the answer, if it has a clean structural form, becomes a tool: any time you have a dense graph and would like to extract a clique from it, the contrapositive of this bound is what does the extraction.

## Background

The relevant objects are simple finite graphs $G=(V,E)$ with $|V|=n$ and $|E|$ edges. The degree of a vertex $v$ is $d(v)$, its number of neighbours; $\sum_v d(v) = 2|E|$ (handshake / double counting). A *clique* $K_p$ is a set of $p$ pairwise-adjacent vertices. An *independent set* is a set of pairwise-non-adjacent vertices. The *clique number* $\omega(G)$ is the size of the largest clique. A graph is $K_{r+1}$-free exactly when $\omega(G)\le r$.

A natural family of $K_{r+1}$-free graphs is the **complete multipartite** graphs. Partition $V$ into $r$ pairwise-disjoint classes $V_1,\dots,V_r$ with $|V_i|=n_i$, $\sum n_i = n$, and join two vertices **iff** they lie in different classes. Call this graph $K_{n_1,\dots,n_r}$. Each class is an independent set; any clique meets each class in at most one vertex, so the largest clique has at most $r$ vertices — such a graph is $K_{r+1}$-free by construction. Its edge count is the number of cross-class pairs,
$$
e(K_{n_1,\dots,n_r}) \;=\; \sum_{i<j} n_i n_j \;=\; \tfrac12\Big(n^2 - \sum_{i=1}^r n_i^2\Big),
$$
the second form because $\big(\sum n_i\big)^2 = \sum n_i^2 + 2\sum_{i<j} n_i n_j$.

Maximizing edges over this family is the same as **minimizing $\sum_i n_i^2$** subject to $\sum_i n_i = n$. Convexity (equivalently Cauchy–Schwarz, $\sum n_i^2 \ge n^2/r$) says the sum of squares is smallest when the parts are as equal as possible. Concretely, if two parts satisfy $n_1 \ge n_2 + 2$, shifting one vertex from the first to the second changes the cross-pair count by $(n_1-1)(n_2+1) - n_1 n_2 = n_1 - n_2 - 1 \ge 1 > 0$, so edges strictly increase. The balanced complete $r$-partite graph — all parts of size $\lfloor n/r\rfloor$ or $\lceil n/r\rceil$, differing by at most one — is therefore the densest member of the family. Writing $n = qr + s$ with $0\le s<r$, it has $s$ parts of size $q+1$ and $r-s$ parts of size $q$; when $r\mid n$ ($s=0$) its edge count is $\big(1-\tfrac1r\big)\tfrac{n^2}{2}$.

A standard structural fact about complete multipartite graphs, definitional and knowable before any extremal theorem: $G$ is complete multipartite **iff non-adjacency is an equivalence relation on $V$** — the classes are then exactly the non-adjacency classes. Reflexivity and symmetry hold for any graph; only transitivity carries content.

## Baselines

**Mantel (1907).** The triangle-free case, $r=2$. A simple graph on $n$ vertices with no $K_3$ has at most $\lfloor n^2/4\rfloor$ edges, attained uniquely by the balanced complete bipartite graph $K_{\lfloor n/2\rfloor,\lceil n/2\rceil}$. The standard argument is $r=2$-specific: triangle-freeness means two adjacent vertices have no common neighbour, so $d(u)+d(v)\le n$ for every edge $uv$; summing over edges and applying an averaging/Cauchy–Schwarz step yields $|E|\le n^2/4$.

**A peeling / induction approach.** Take a $K_{r+1}$-free graph $G$ with the maximum number of edges. Locate within it a high-degree vertex $x$ (or a $K_r$), split $V$ around it, and bound the edges in three groups — inside the chosen set, between it and the rest, and inside the rest — using a $K_r$-free or $K_{r+1}$-free bound on the smaller pieces, then close by induction on $n$.

**An optimization / Lagrangian approach.** Place a probability weight $w_i\ge 0$ on each vertex, $\sum_i w_i = 1$, and maximize $f(w)=\sum_{v_iv_j\in E} w_i w_j$. Moving all of one vertex's weight onto a non-adjacent neighbour-of-larger-weighted-sum never decreases $f$, so the optimum concentrates on a clique; on a $k$-clique with uniform weights $f=\tfrac12(1-1/k)$, increasing in $k$, capped at $k=r$ since $\omega(G)\le r$. Evaluating $f$ at the uniform distribution $w_i=1/n$ gives $|E|/n^2 = f \le \tfrac12(1-1/r)$, i.e. the bound.

**A probabilistic / counting approach.** A random vertex ordering and a greedy clique built along it has expected size $\sum_i 1/(n-d_i)$, so $\omega(G)\ge \sum_i 1/(n-d_i)$; combining with Cauchy–Schwarz and $\omega(G)\le r$ and $\sum_i d_i = 2|E|$ yields $n^2 \le r\,(n^2 - 2|E|)$, again the bound.

## Evaluation settings

This is a theorem, so the "evaluation" is mathematical rather than empirical. The yardsticks that exist in advance:

- **Correctness and generality.** Does the argument hold for every $n$ and every $r\ge 1$ (with $r=2$ specializing to the known Mantel bound), and is each step valid, including the boundary cases $n<r$?
- **Strength of the conclusion.** Does it yield not only the exact edge maximum but also the *full set of extremal graphs* with uniqueness?
- **Sanity on small cases.** For small $n$ the entire space of $K_{r+1}$-free graphs can be enumerated by brute force, the maximum edge count read off directly, and compared against the closed-form prediction $\tfrac12(n^2-\sum n_i^2)$ for the balanced partition. The two operations a structural proof would use can likewise be applied to a small explicit graph and checked to (a) preserve $K_{r+1}$-freeness and (b) move the edge count in the claimed direction. These checks use only enumeration and basic graph predicates that exist independently of any theorem.
- **Economy.** Among correct proofs, the natural secondary criterion is directness — how little machinery is invoked.

## Code framework

No heavy machinery is needed; the only code that is natural here is a small finite-$n$ sanity checker built from primitives that already exist (enumeration over subsets, an adjacency representation, a clique predicate). It is a *cross-check* on a structural claim, not the claim itself. Pre-method scaffold:

```python
from itertools import combinations, product

def parts_balanced(n, r):
    """Sizes of r parts as equal as possible (each floor(n/r) or ceil(n/r))."""
    q, s = divmod(n, r)
    return [q + 1] * s + [q] * (r - s)

def candidate_edge_count(n, r):
    """Edge count of the densest complete r-partite graph: (n^2 - sum sizes^2)/2."""
    sizes = parts_balanced(n, r)
    return (n * n - sum(a * a for a in sizes)) // 2

def has_clique(adj, k):
    """True iff the graph (set of frozenset edges) contains a K_k."""
    # TODO: enumerate k-subsets, test all pairs adjacent
    pass

def brute_force_max_edges(n, r):
    """Exhaustive max |E| over all K_{r+1}-free graphs on n labelled vertices."""
    # TODO: iterate over every subset of the C(n,2) possible edges,
    #       keep those with no K_{r+1}, return the largest edge count
    pass

def local_move(adj, n, *args):
    """A local transformation on the graph, to be designed.

    # TODO: the operation the argument will turn on.
    """
    pass

if __name__ == "__main__":
    # cross-check the candidate count against brute force on small n, r
    for n in range(2, 8):
        for r in range(1, n):
            assert candidate_edge_count(n, r) == brute_force_max_edges(n, r)
    # apply local_move to a small explicit graph and check it preserves
    # K_{r+1}-freeness and does not lose edges
```

The body of `local_move` is the empty slot the structural argument will fill; `has_clique` and `brute_force_max_edges` are generic predicates over an explicit graph.
