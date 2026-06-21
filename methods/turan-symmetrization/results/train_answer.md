# Turán's theorem, via symmetrization

Fix $n$ vertices and forbid a clique on $r+1$ of them — no $K_{r+1}$ anywhere in the graph. I want two things, not one: the exact maximum number of edges such a graph can carry, as a function of $n$ and $r$, and — the harder and more interesting half — a complete description of *which* graphs attain that maximum, with uniqueness. A bound on the count alone tells me how dense a $K_{r+1}$-free graph can be but not what a maximally dense one must look like, and it is the structure that turns the result into a tool: whenever I face a dense graph and want to pull a clique out of it, it is the contrapositive of a *structural* extremal theorem that does the extracting. So a solution must be exact, hold for all $n$ and all $r$, and pin down the extremizer.

The one fully satisfying precedent is Mantel's $r=2$ case: a triangle-free graph on $n$ vertices has at most $\lfloor n^2/4\rfloor$ edges, uniquely the balanced complete bipartite graph. That answer is suggestive — the densest triangle-free graph is the *most symmetric* object one could draw, two equal classes with every cross-edge and no inside-edge — but Mantel's proof is $r=2$-specific. It rests on the fact that two adjacent vertices share no common neighbour, so $d(u)+d(v)\le n$ on every edge; for a forbidden $K_{r+1}$ with $r\ge 3$ there is no comparably clean per-edge degree inequality, and the method stalls. The other routes I know reach general $r$ but each falls short of what I want. Induction by peeling — fix a $K_r$, split the rest around it, bound edges inside, across, and beyond, and close on $n$ — is correct but it is bookkeeping: three coupled edge-counts and an inductive bound all forced tight at the end, and the multipartite shape of the extremizer only crawls out as the configuration where every inequality happens to be simultaneously tight. The shape is *recovered*, never *used*. The Lagrangian weighting trick ($\max\sum_{ij\in E} w_iw_j$ over a probability vector, concentrating onto a clique of size $\le r$) and the random-greedy-clique counting argument ($\omega(G)\ge\sum_i 1/(n-d_i)$, combined with Cauchy–Schwarz) are both short and elegant, but each delivers only the *inequality* and says nothing about which graphs are extremal or whether the optimum is unique. The shared gap is a single direct mechanism that produces the exact maximum *and* the unique extremal structure together, for all $n$ and all $r$.

So I stop trying to bound the count and decide the real content is *shape*, not number. The candidate is clear: partition $V$ into $r$ classes and join two vertices exactly when they lie in different classes — the complete $r$-partite graph. It is $K_{r+1}$-free for free, since a clique can take at most one vertex per class, so $\omega\le r$ regardless of class sizes. Its edge count is the number of cross-class pairs, $\sum_{i<j}n_in_j=\tfrac12(n^2-\sum_i n_i^2)$ by the identity $(\sum n_i)^2=\sum n_i^2+2\sum_{i<j}n_in_j$ with $\sum n_i=n$; so the count depends on the class sizes only through $\sum_i n_i^2$, and maximizing edges means *minimizing* that sum of squares. A vertex-shifting move settles the minimization: if $n_1\ge n_2+2$, moving one vertex from class $1$ to class $2$ changes the edge count by $(n_1-1)(n_2+1)-n_1n_2=n_1-n_2-1\ge 1>0$, strictly up, so the densest member has all classes as equal as possible. Call that balanced graph the **Turán graph** $T(n,r)$ — $r$ classes of size $\lfloor n/r\rfloor$ or $\lceil n/r\rceil$, differing by at most one. That is an immediate *lower* bound on the answer; the whole problem is the matching upper bound and uniqueness.

I propose to prove Turán's theorem by **Zykov symmetrization**: rather than bound $e(G)$ directly, show that any *edge-maximal* $K_{r+1}$-free graph is forced to be complete multipartite, then read off $T(n,r)$. The bridge is a structural restatement: a graph is complete multipartite if and only if non-adjacency is an equivalence relation on $V$ — and reflexivity and symmetry are automatic, so the only content is transitivity. Non-adjacency fails to be transitive exactly when there is a *guilty triple* $u,v,w$ with $uv\notin E$, $uw\notin E$, but $vw\in E$: $u$ sits apart from both $v$ and $w$, yet $v$ and $w$ are joined. If a maximal $G$ were not complete multipartite, such a triple would exist, and I will manufacture from it a $K_{r+1}$-free graph with strictly more edges — a contradiction.

What makes the argument go is choosing a local surgery that is both clique-safe and edge-monotone, and **vertex duplication** is the one operation with both properties built in. To *duplicate* a vertex $v$ is to give some vertex exactly the neighbourhood $N(v)$ and make it non-adjacent to $v$. Since $v$ and its clone are non-adjacent twins with identical neighbourhoods, no clique can contain both (they are not joined), and any clique through the clone has a mirror clique through $v$ (same neighbours); hence duplication never raises the clique number and preserves $K_{r+1}$-freeness. Its effect on the count is trivial to track with no algebra: deleting a vertex $u$ and re-attaching it as a clone of $v$ removes $u$'s $d(u)$ edges and adds a fresh copy of $v$'s $d(v)$ edges, a net change of $d(v)-d(u)$. So I should clone the *higher-degree* end and come out ahead.

Apply it to the guilty triple, where $u$ misses both $v$ and $w$. If $u$ is not the largest — $d(u)<d(v)$ (or symmetrically $d(u)<d(w)$) — delete $u$ and re-create it as a clone of $v$: still $K_{r+1}$-free, and the count rises by $d(v)-d(u)>0$. The wall is the other case: what if $u$ is itself the high-degree vertex, $d(u)\ge d(v)$ and $d(u)\ge d(w)$? Then a single clone of $v$ over $u$ gives $d(v)-d(u)\le 0$ — no gain. The fix is to notice the configuration is asymmetric in a way I can exploit: $u$ misses *two* vertices, and those two are joined. So I use $u$'s big degree *twice* — delete both $v$ and $w$ and put *two* clones of $u$ in their place (each carrying $N(u)$, mutually non-adjacent and non-adjacent to $u$), still $K_{r+1}$-free by the same twin argument. The count, with the one subtlety that is the heart of the proof: deleting $v$ removes $d(v)$ edges, but deleting $w$ then removes only $d(w)-1$ further edges, because the edge $vw$ already vanished when $v$ went — it must not be double-counted. The two clones add $2d(u)$. So the net change is $2d(u)-(d(v)+d(w)-1)=\big(2d(u)-d(v)-d(w)\big)+1$, and in this case $2d(u)\ge d(v)+d(w)$, so the change is $\ge 1>0$, strict. The very feature that stalled the single clone — $u$ being the largest — is what powers the double clone: I cash in $u$'s degree twice and pay only for the two smaller degrees, minus the shared edge $vw$ I get to drop for free. That $+1$ is exactly the edge $vw$, which is why strictness survives even at the boundary $2d(u)=d(v)+d(w)$, and it is precisely the hypothesis that made the triple guilty. The two cases are exhaustive — the negation of "$d(u)<d(v)$ or $d(u)<d(w)$" is "$d(u)\ge d(v)$ and $d(u)\ge d(w)$" — so every guilty triple yields a strict improvement, and a maximal graph has none: non-adjacency is transitive and $G$ is complete multipartite, $G=K_{n_1,\dots,n_m}$.

From there the rest reads off with no induction. There cannot be $m\ge r+1$ nonempty classes, since one vertex from each of $r+1$ classes is pairwise across-class, hence a $K_{r+1}$; so $m\le r$, and I take $m=r$ allowing empty classes, because merging two classes only deletes the cross-edges that ran between them. Then the vertex-shifting computation forces the classes balanced to minimize $\sum_i n_i^2$, giving $G=T(n,r)$; and since $\sum_i n_i^2\ge n^2/r$ with equality iff balanced, every $K_{r+1}$-free graph obeys $e(G)\le\tfrac12(n^2-n^2/r)=(1-\tfrac1r)\tfrac{n^2}{2}$. No step had any slack — a maximal graph that is not $T(n,r)$ always admits a strictly improving move — so $T(n,r)$ is the *unique* extremizer. Setting $r=2$ recovers Mantel's $\lfloor n^2/4\rfloor$ and its balanced bipartite extremizer as the first case, and the boundary $n<r$ is consistent: the balanced graph is then $K_n$, which is $K_{r+1}$-free and carries all $\binom n2$ edges. With $n=qr+s$, $0\le s<r$, the exact count is

$$
e(T(n,r)) \;=\; \tfrac12\Big(n^2-\textstyle\sum_i n_i^2\Big) \;=\; \big(1-\tfrac1r\big)\frac{n^2-s^2}{2}+\binom{s}{2},
$$

which equals $\big(1-\tfrac1r\big)\dfrac{n^2}{2}$ when $r\mid n$, and $\big\lfloor(1-\tfrac1r)\tfrac{n^2}{2}\big\rfloor$ for $r\le 7$.

The finished statement and proof are these. Let $G$ be a simple graph on $n$ vertices with no $K_{r+1}$ (i.e. $\omega(G)\le r$). Then

$$
e(G)\;\le\; e\big(T(n,r)\big)\;\le\;\Big(1-\tfrac1r\Big)\frac{n^2}{2},
$$

with $e(G)=e(T(n,r))$ if and only if $G\cong T(n,r)$. For $r=2$ this is Mantel's theorem: a triangle-free graph has at most $\lfloor n^2/4\rfloor$ edges, uniquely the balanced complete bipartite graph.

*Proof (Zykov symmetrization).* Let $G$ be a $K_{r+1}$-free graph on $n$ vertices with the maximum number of edges.

**Step 1 — an edge-maximal $K_{r+1}$-free graph is complete multipartite.** A graph is complete multipartite iff non-adjacency is an equivalence relation on $V$, i.e. iff non-adjacency is transitive. Suppose not: there exist vertices $u,v,w$ with

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

No step had slack: a maximal graph that is not $T(n,r)$ admits a strictly improving move, so $T(n,r)$ is the unique extremizer. $\qquad\blacksquare$

For my own peace of mind I would watch this on tiny cases — enumerate every $K_{r+1}$-free graph on a handful of vertices, read off the true maximum, and confirm it equals the closed form for the balanced split; and take one explicit small graph with a guilty triple, apply the clone move, and check the edge count rose while no $K_{r+1}$ appeared. Nothing in the proof needs it, but it is the check I would run the moment I had a machine.

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
