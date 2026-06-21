# Turán's theorem, via symmetrization

## Problem

Among all simple graphs on $n$ vertices containing no clique $K_{r+1}$, determine the maximum number of edges and the graphs that attain it.

## The extremal graph

For $r\ge 1$, the **Turán graph** $T(n,r)$ is the complete $r$-partite graph whose $r$ classes are as equal as possible (each of size $\lfloor n/r\rfloor$ or $\lceil n/r\rceil$): join two vertices iff they lie in different classes. It is $K_{r+1}$-free because any clique meets each class in at most one vertex, so $\omega(T(n,r))\le r$. Writing $n=qr+s$ with $0\le s<r$ (so $s$ classes of size $q+1$ and $r-s$ of size $q$), its edge count is
$$
e(T(n,r)) \;=\; \tfrac12\Big(n^2-\textstyle\sum_i n_i^2\Big) \;=\; \big(1-\tfrac1r\big)\frac{n^2-s^2}{2}+\binom{s}{2},
$$
which equals $\big(1-\tfrac1r\big)\dfrac{n^2}{2}$ when $r\mid n$, and $\big\lfloor(1-\tfrac1r)\tfrac{n^2}{2}\big\rfloor$ for $r\le 7$.

## Key idea

Do not bound the edge count directly. Instead show that any **edge-maximal** $K_{r+1}$-free graph is forced to be complete multipartite, using one local, clique-safe, edge-monotone operation — **vertex duplication** (cloning a vertex's neighbourhood). Then balancing the classes (a convexity step) pins down $T(n,r)$ uniquely, and the edge count follows. The same argument yields the inequality and the uniqueness of the extremizer in one stroke, for all $n$ and all $r$.

The operation: to *duplicate* a vertex $v$ is to give some vertex exactly the neighbourhood $N(v)$, non-adjacent to $v$. Since $v$ and its clone are non-adjacent twins, no clique uses both, and any clique through the clone has a copy through $v$; hence duplication never raises the clique number — it preserves $K_{r+1}$-freeness. Re-attaching a vertex $u$ as a clone of $v$ changes the edge count by $d(v)-d(u)$.

## Theorem

Let $G$ be a simple graph on $n$ vertices with no $K_{r+1}$ (i.e. $\omega(G)\le r$). Then
$$
e(G)\;\le\; e\big(T(n,r)\big)\;\le\;\Big(1-\tfrac1r\Big)\frac{n^2}{2},
$$
with $e(G)=e(T(n,r))$ if and only if $G\cong T(n,r)$. For $r=2$ this is Mantel's theorem: a triangle-free graph has at most $\lfloor n^2/4\rfloor$ edges, uniquely the balanced complete bipartite graph.

## Proof (Zykov symmetrization)

Let $G$ be a $K_{r+1}$-free graph on $n$ vertices with the **maximum** number of edges.

**Step 1 — an edge-maximal $K_{r+1}$-free graph is complete multipartite.**
A graph is complete multipartite iff non-adjacency is an equivalence relation on $V$, i.e. iff non-adjacency is transitive. Suppose not: there exist vertices $u,v,w$ with
$$
uv\notin E,\qquad uw\notin E,\qquad vw\in E.
$$

*Case 1: $d(u)<d(v)$ (or symmetrically $d(u)<d(w)$).* Delete $u$ and re-attach it as a clone of $v$ (neighbourhood $N(v)$, non-adjacent to $v$). The result $G'$ is $K_{r+1}$-free (duplication preserves clique number) and
$$
e(G') = e(G) + d(v) - d(u) > e(G),
$$
contradicting maximality.

*Case 2: $d(u)\ge d(v)$ and $d(u)\ge d(w)$.* Delete both $v$ and $w$ and duplicate $u$ twice (two clones of $u$, mutually non-adjacent and non-adjacent to $u$). Deleting $v$ removes $d(v)$ edges; deleting $w$ then removes only $d(w)-1$ further edges, since the edge $vw$ vanished with $v$. The two clones add $2d(u)$ edges. So $G''$ is $K_{r+1}$-free and
$$
e(G'') = e(G) + 2d(u) - \big(d(v)+d(w)-1\big) \;=\; e(G) + \underbrace{\big(2d(u)-d(v)-d(w)\big)}_{\ge 0} + 1 \;>\; e(G),
$$
again contradicting maximality. (The strictness uses the $+1$, i.e. that $vw\in E$.)

The two cases are exhaustive, so no such triple exists: non-adjacency is transitive and $G$ is complete multipartite, $G=K_{n_1,\dots,n_m}$.

**Step 2 — at most $r$ classes.** If $m\ge r+1$, one vertex from each of $r+1$ classes forms a $K_{r+1}$. So $m\le r$; take $m=r$, allowing empty classes (merging classes only deletes cross-edges, so using all $r$ classes is at least as good).

**Step 3 — balance the classes.** For a complete $r$-partite graph, $e=\tfrac12\big(n^2-\sum_i n_i^2\big)$, so maximizing edges means minimizing $\sum_i n_i^2$ subject to $\sum_i n_i=n$. If $n_1\ge n_2+2$, moving one vertex from class $1$ to class $2$ changes the edge count by $(n_1-1)(n_2+1)-n_1n_2 = n_1-n_2-1\ge 1>0$. Hence the maximum requires $|n_i-n_j|\le 1$ for all $i,j$, i.e. $G=T(n,r)$, and (since $\sum_i n_i^2\ge n^2/r$ with equality iff balanced) every $K_{r+1}$-free graph obeys $e(G)\le \tfrac12(n^2-n^2/r)=(1-\tfrac1r)\tfrac{n^2}{2}$.

No step had slack: a maximal graph that is not $T(n,r)$ admits a strictly improving move, so $T(n,r)$ is the **unique** extremizer. $\qquad\blacksquare$

## Finite-$n$ verification (optional cross-check)

A small script confirms the closed form equals the exhaustive maximum for tiny $(n,r)$, and that one clone step strictly adds edges while preserving $K_{r+1}$-freeness. It is a sanity check, not part of the proof.

```python
from itertools import combinations, product

def parts_balanced(n, r):
    # balanced r classes: s of size q+1, r-s of size q, where n = qr + s
    q, s = divmod(n, r)
    return [q + 1] * s + [q] * (r - s)

def turan_edges(n, r):
    # e(T(n,r)) = (n^2 - sum |V_i|^2) / 2
    sizes = parts_balanced(n, r)
    return (n * n - sum(a * a for a in sizes)) // 2

def adj_n(adj):
    m = 0
    for e in adj:
        m = max(m, max(e) + 1)
    return m

def has_clique(adj, k):
    n = adj_n(adj)
    for S in combinations(range(n), k):
        if all(frozenset((u, v)) in adj for u, v in combinations(S, 2)):
            return True
    return False

def brute_force_max_edges(n, r):
    pairs = [frozenset(p) for p in combinations(range(n), 2)]
    best = -1
    for bits in product((0, 1), repeat=len(pairs)):
        adj = {e for e, b in zip(pairs, bits) if b}
        if not has_clique(adj, r + 1):
            best = max(best, len(adj))
    return best

def clone(adj, n, keep, drop):
    # delete 'drop', re-attach as a twin of 'keep' (neighbourhood of keep, not adjacent to keep)
    keep_nbrs = {u for u in range(n) if u != keep and frozenset((u, keep)) in adj}
    new = {e for e in adj if drop not in e}
    for u in keep_nbrs:
        if u != drop:
            new.add(frozenset((u, drop)))
    return new

if __name__ == "__main__":
    for n in range(2, 8):                       # closed form == exhaustive maximum
        for r in range(1, n):
            assert turan_edges(n, r) == brute_force_max_edges(n, r)
    n, r = 4, 2                                 # clone step on a guilty triple
    G = {frozenset((0, 1)), frozenset((1, 2)), frozenset((2, 3))}
    H = clone(G, n, keep=2, drop=0)            # 0,2 non-adjacent, deg(2) > deg(0)
    assert not has_clique(H, r + 1) and len(H) > len(G)
```
