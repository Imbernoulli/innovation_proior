# Reasoning

I have the problem in front of me and almost nothing else — no desk, no paper to spread out, just a long shift walking among stacked logs and a question I can turn over in my head. That constraint is going to shape everything: whatever I land on has to be something I can *hold*, a structural picture, not a page of algebra I'd lose track of by the third line. So let me think it through from the picture.

The question. A graph on $n$ vertices, and I forbid a clique on $r+1$ vertices — no $K_{r+1}$ anywhere. How many edges can I cram in, and what does the densest such graph look like? Mantel already did $r=2$: forbid a triangle and you top out at $\lfloor n^2/4\rfloor$, achieved by splitting the vertices into two equal halves and joining everything across the split. That answer is suggestive in a way I want to take seriously. The extremal triangle-free graph isn't some intricate sparse thing — it's the *most symmetric* object you could draw: two classes, all cross-edges, no inside-edges. Beauty and extremality coincided there. I have a dim feeling they will again.

So before I prove anything, let me guess the shape and see if it's even plausible. Generalize Mantel's two classes to $r$ classes. Partition the vertices into $r$ groups $V_1,\dots,V_r$, put an edge between two vertices exactly when they're in *different* groups, and never inside a group. Call it $K_{n_1,\dots,n_r}$ with $|V_i|=n_i$. Is it $K_{r+1}$-free? A clique can grab at most one vertex from each class — two vertices in the same class are non-adjacent, so they can't both be in a clique. There are $r$ classes, so the biggest clique has at most $r$ vertices. Good: no $K_{r+1}$, automatically, for free, no matter how I size the classes. This whole family is admissible.

How many edges does it have? It's all the cross-class pairs. Total pairs of vertices is $\binom n2$, but I only keep the ones straddling two classes; I throw away the within-class pairs. Cleaner to count directly: the number of cross pairs is $\sum_{i<j} n_i n_j$. And there's a tidy identity for that — $\big(\sum_i n_i\big)^2 = \sum_i n_i^2 + 2\sum_{i<j} n_i n_j$, and $\sum_i n_i = n$, so
$$
e(K_{n_1,\dots,n_r}) = \sum_{i<j} n_i n_j = \tfrac12\Big(n^2 - \sum_i n_i^2\Big).
$$
Lovely — the edge count depends on the class sizes *only through $\sum_i n_i^2$*, and to maximize edges I want that sum of squares as *small* as possible, with $\sum_i n_i = n$ fixed.

Sum of squares small under a fixed sum — that's a convexity statement, the parts want to be equal. Let me not invoke a theorem I can't write down here; let me just check it with a move I can do in my head. Suppose two of the classes are lopsided, say $n_1 \ge n_2 + 2$. Take one vertex out of $V_1$ and drop it into $V_2$. The cross-pair count changes by $(n_1-1)(n_2+1) - n_1 n_2$. Expand: $n_1 n_2 + n_1 - n_2 - 1 - n_1 n_2 = n_1 - n_2 - 1 \ge 1 > 0$. So moving a vertex from the bigger class to the smaller one *strictly* adds edges whenever the classes differ by two or more. I can keep doing this until no two classes differ by more than one. So within the multipartite family, the densest one has all classes as equal as possible — sizes $\lfloor n/r\rfloor$ or $\lceil n/r\rceil$. If $r$ divides $n$ exactly, each class is $n/r$ and the edge count is $\tfrac12\big(n^2 - r(n/r)^2\big) = \tfrac12 n^2(1 - 1/r) = \big(1-\tfrac1r\big)\tfrac{n^2}{2}$.

So I have a concrete candidate — the balanced $r$-partite graph — and an easy *lower* bound on the answer: the true maximum is at least this many edges. The whole problem is now the *upper* bound. Could some craftier $K_{r+1}$-free graph, one that *isn't* built by splitting into classes, beat my balanced multipartite graph? My instinct says no — symmetry should win — but instinct isn't a proof.

Let me try the obvious thing first: induction on $n$, the way I'd attack most extremal problems. Take a $K_{r+1}$-free graph $G$ with the most edges. It must contain a $K_r$ somewhere — if it had no $K_r$ at all I could keep adding edges without making any $K_{r+1}$, contradicting maximality. Call that clique $A$, and $B$ the other $n-r$ vertices. Inside $A$: $\binom r2$ edges. Each vertex of $B$ can be adjacent to at most $r-1$ of the $r$ vertices of $A$ — if it were adjacent to all $r$, it would complete a $K_{r+1}$. So edges between $A$ and $B$ number at most $(r-1)(n-r)$. And edges inside $B$: $B$ is itself $K_{r+1}$-free on $n-r$ vertices, so by induction it has at most as many edges as the balanced multipartite graph on $n-r$ vertices. Add the three bounds.

That *works*, it's correct — I keep starting down it in the copybook in my head — but it keeps breaking when I try to make it clean. The trouble is the three pieces don't tighten at the same configuration without me carefully forcing each inequality to equality, and worst of all the *structure* — the thing I actually care about, that the extremizer is multipartite and balanced — only crawls out at the very end, as the configuration where all three bounds happen to be tight simultaneously. I'm computing my way *toward* the multipartite shape instead of arguing *about* it. And there's the practical wall: tracking three coupled edge-counts and an inductive bound is exactly the kind of bookkeeping I can't keep straight without paper. The various approaches by induction all start promisingly and then stall in the same swamp. I need something I can carry in my head — a structural argument, not an accounting one.

So back up. What do I actually believe? I believe the extremal graph *is* one of my multipartite graphs. If I could prove that — that any edge-maximal $K_{r+1}$-free graph must already be complete multipartite — then I'd be done in two strokes: pick $r$ as the number of classes (more than $r$ classes would force a $K_{r+1}$; fewer would just be a special case with empty classes), and balance the classes by the vertex-shifting move I already did. The hard, structural half is: *why must the optimum be complete multipartite at all?* Let me make that the whole target and forget edge-counting until the end.

What does "complete multipartite" even mean, structurally, in a way I can grab? It means the vertices fall into classes, within-class non-adjacent, across-class adjacent. That's exactly the statement that "being non-adjacent" sorts the vertices into clean groups — that non-adjacency is an *equivalence relation*. It's reflexive and symmetric for free. The only thing that can fail is transitivity. So $G$ is complete multipartite if and only if non-adjacency is transitive: whenever $u$ is non-adjacent to $v$ and $u$ is non-adjacent to $w$, then $v$ and $w$ are non-adjacent too.

Flip that around to see what *breaks* multipartiteness. Non-adjacency fails to be transitive exactly when there's a triple $u,v,w$ with $u$ non-adjacent to $v$, $u$ non-adjacent to $w$, *but* $v$ adjacent to $w$. Picture it: $u$ sits apart from both $v$ and $w$, yet $v$ and $w$ are joined. If my graph is *not* complete multipartite, such a guilty triple exists. So suppose $G$ is edge-maximal and $K_{r+1}$-free but, for contradiction, *not* complete multipartite — then I have this triple $u,v,w$ in hand: $uv\notin E$, $uw\notin E$, $vw\in E$. I want to manufacture from it another $K_{r+1}$-free graph with *more* edges, contradicting that $G$ was maximal. If I can always do that, then maximal forces multipartite.

Now, what local surgery can I do that I can *trust* not to create a $K_{r+1}$? Adding edges is dangerous — any edge I throw in might complete a forbidden clique. Deleting edges is safe but loses me edges, the wrong direction. I need a move that's both clique-safe and edge-monotone. Here's the one operation that has both properties built in: take a vertex, and *clone* its neighbourhood onto another vertex. Concretely, "duplicating" a vertex $v$ means: make some other vertex carry exactly $v$'s neighbours — the same set $N(v)$ — and make sure that clone is *not* adjacent to $v$ itself. Why is that safe? Because $v$ and its clone are non-adjacent twins with identical neighbourhoods. No clique can contain both of them — they're not joined. And any clique that uses the clone could equally well use $v$ in its place, since they see the same neighbours; so the clone never lets me build a clique bigger than one already sitting on $v$. If $G$ had no $K_{r+1}$, neither does the graph after duplication. The clique number can't go up. That's the operation I can lean on.

And its effect on edges is dead simple to compute, which is exactly what I want with no paper: if I take the vertex $u$ and *replace* its current connections with a clone of $v$'s connections — delete $u$ as it stands and re-attach it as a copy of $v$ — I lose the $d(u)$ edges $u$ had and gain the $d(v)$ edges of a fresh copy of $v$. The net change is $d(v) - d(u)$. So: clone the *higher-degree* one of the two, and I come out ahead.

Let me try it on the guilty triple. I have $uv\notin E$ and $uw\notin E$ — so $u$ is non-adjacent to both $v$ and $w$. Compare degrees. Say first that $d(u) < d(v)$ (or symmetrically $d(u)<d(w)$; same thing). Then delete $u$ and re-create it as a clone of $v$: a vertex with exactly $v$'s neighbourhood, non-adjacent to $v$. The new graph is still $K_{r+1}$-free, because cloning preserves clique number. And the edge count changed by
$$
\Delta = d(v) - d(u) > 0.
$$
More edges, still $K_{r+1}$-free — that contradicts $G$ being edge-maximal.

So that case is killed whenever *some* endpoint of the two non-edges at $u$ has higher degree than $u$. But — wait, stare at this — what if it doesn't? What if $u$ is the *high*-degree vertex of the triple, $d(u) \ge d(v)$ *and* $d(u) \ge d(w)$? Then cloning $v$ over $u$ gives $\Delta = d(v) - d(u) \le 0$ — could even be zero — no contradiction. The single clone is stuck. This is the wall. I can't just always clone the bigger neighbour, because sometimes the bigger vertex is the lonely one $u$ that sits apart from the other two.

Sit with that stuck case. $u$ is non-adjacent to both $v$ and $w$, and $u$ has the largest degree of the three. The thing $u$ has going for it is precisely that high degree — and I'm only using it once when I clone. What if I use it *twice*? The configuration is asymmetric in a useful way: $u$ misses *two* vertices, $v$ and $w$, and those two are joined to each other. So let me delete *both* $v$ and $w$ and put *two* clones of $u$ in their place — duplicate $u$ twice. Two fresh copies of $u$, each carrying $u$'s neighbourhood, each non-adjacent to $u$ and (I'll make them) non-adjacent to each other. Still $K_{r+1}$-free: same twin argument, the clones can't extend any clique beyond what sits on $u$.

Now count the edge change, carefully, because this is where a sign slip would wreck it. I delete $v$: that removes the $d(v)$ edges at $v$. I delete $w$: that removes $w$'s edges — but one of $w$'s edges, the edge $vw$, is *already gone*, because I just deleted $v$. So deleting $w$ removes only $d(w) - 1$ further edges. Together I've removed $d(v) + d(w) - 1$ edges. Then I add two clones of $u$: each clone brings $d(u)$ edges, so $+2d(u)$. Net:
$$
\Delta = 2d(u) - \big(d(v) + d(w) - 1\big) = 2d(u) - d(v) - d(w) + 1.
$$
And in this stuck case $d(u) \ge d(v)$ and $d(u) \ge d(w)$, so $2d(u) \ge d(v) + d(w)$, hence $\Delta \ge 1 > 0$. Strictly more edges. Still $K_{r+1}$-free. Contradiction again. The very feature that stalled the single clone — $u$ being the high-degree vertex — is what makes the double clone win: I cash in $u$'s big degree twice and only pay for the two smaller degrees, minus the one shared edge $vw$ I get to drop for free.

That $+1$ is doing real work and I want to be sure it's honest. It's the edge $vw$. It's the single edge of the guilty triple. If I'd carelessly written the cost as $d(v)+d(w)$, I'd be double-removing that edge — counting it once in $d(v)$ and again in $d(w)$ — when physically it's one edge that disappears the moment its first endpoint goes. The corrected cost is $d(v)+d(w)-1$, and that correction is exactly why the inequality is *strict* even when $2d(u) = d(v)+d(w)$. Without the $vw$ edge being present I'd have no guarantee of strictness in the boundary case. So it matters that $v$ and $w$ are *adjacent* — which is precisely the hypothesis that made the triple guilty in the first place. The whole thing closes on itself.

Let me make sure I've covered the cases exhaustively. Given the guilty triple $u,v,w$ ($u$ non-adjacent to $v$ and to $w$, and $vw$ an edge): either $u$ is *not* the unique top degree — i.e. $d(u)<d(v)$ or $d(u)<d(w)$ — and the single clone of the bigger one wins; or $u$ *is* at least as big as both — $d(u)\ge d(v)$ and $d(u)\ge d(w)$ — and the double clone of $u$ wins. Those two cases are exhaustive (the negation of "$d(u)<d(v)$ or $d(u)<d(w)$" is exactly "$d(u)\ge d(v)$ and $d(u)\ge d(w)$"). Every guilty triple yields a strictly-better $K_{r+1}$-free graph. Therefore an edge-maximal $K_{r+1}$-free graph has *no* guilty triple: non-adjacency is transitive, an equivalence relation, and $G$ is complete multipartite. That's the whole structural fact, and I held it in my head with two pictures and one subtraction.

Now I just read off the rest, no induction needed. $G$ is complete multipartite, $G = K_{n_1,\dots,n_m}$ for some classes. How many classes $m$? It can't have $m \ge r+1$ nonempty classes — pick one vertex from each of $r+1$ different classes and they're pairwise across-class, hence pairwise adjacent, hence a $K_{r+1}$, forbidden. So $m \le r$. And having fewer than $r$ classes only loses edges: I'd rather split into $r$ classes and use all of them (formally, take $m=r$ and allow some classes to be empty if $n$ is small; empty classes contribute nothing, and any nonempty graph on $n\ge r$ vertices does strictly better with $r$ classes than with fewer, since merging two classes deletes all the edges that had run between them). So the edge-maximal graph is a complete $r$-partite graph, and by the vertex-shifting computation I did at the very start its classes must be balanced — sizes differing by at most one, $\lfloor n/r\rfloor$ or $\lceil n/r\rceil$ — to minimize $\sum n_i^2$. That graph is unique up to relabelling, and the surgery argument never had any slack: at *every* maximal graph the only way to avoid a strictly-improving move was to be exactly this balanced multipartite graph. So it's not just optimal, it's the *unique* optimum.

The edge count drops out. With balanced classes, write $n = qr + s$, $0\le s<r$: that's $s$ classes of size $q+1$ and $r-s$ of size $q$. Plug into $\tfrac12(n^2 - \sum n_i^2)$. When $s=0$ — $r$ divides $n$ — it's $\tfrac12\big(n^2 - r q^2\big) = \tfrac12\big(n^2 - n^2/r\big) = \big(1-\tfrac1r\big)\tfrac{n^2}{2}$. In general $\sum n_i^2 = s(q+1)^2 + (r-s)q^2$, giving the exact integer count; and for the clean upper bound, since $\sum n_i^2 \ge n^2/r$ always (the balanced split minimizes it), every $K_{r+1}$-free graph satisfies
$$
|E| \le \tfrac12\Big(n^2 - \tfrac{n^2}{r}\Big) = \Big(1-\tfrac1r\Big)\tfrac{n^2}{2},
$$
with equality iff $G$ is the balanced complete $r$-partite graph. Set $r=2$ and this is Mantel's $\lfloor n^2/4\rfloor$ with its balanced-bipartite extremizer, recovered as the first case — good, the general argument contains the one I trusted.

Let me sanity-check the boundary $n < r$. Then I can't even fill $r$ classes; the balanced graph is just $r$ classes of size $0$ or $1$, which is the complete graph $K_n$ — and indeed $K_n$ has no $K_{r+1}$ when $n<r+1$, so the maximum is *all* $\binom n2$ edges, consistent. No special pleading needed.

So the path was: distrust the inductive bookkeeping, decide the real content is *shape* not *count*, reduce "extremal" to "complete multipartite," characterize multipartite as "non-adjacency is transitive," locate the local obstruction as a guilty triple, and kill every guilty triple with a clique-safe edge-monotone move — clone the higher-degree end of a non-edge; and when the lonely vertex is itself the high-degree one, clone *it twice* and delete the two it misses, the shared edge between them paying for the strict gain. Maximal therefore forces complete multipartite; multipartite plus $K_{r+1}$-free forces at most $r$ classes; minimizing $\sum n_i^2$ forces them balanced; and the count follows. One picture, carried whole.

For my own peace of mind I'd want to watch this on tiny cases — enumerate every $K_{r+1}$-free graph on a handful of vertices, read off the true maximum, and check it equals $\tfrac12(n^2-\sum n_i^2)$ for the balanced split; and take one explicit small graph with a guilty triple, apply the clone move, and confirm the edge count went up while no $K_{r+1}$ appeared. Nothing in the proof needs it, but it's the kind of check I'd run the moment I had a machine.

```python
from itertools import combinations, product

def parts_balanced(n, r):
    # balanced r classes: s of size q+1, r-s of size q, where n = qr + s
    q, s = divmod(n, r)
    return [q + 1] * s + [q] * (r - s)

def turan_edges(n, r):
    # (n^2 - sum |V_i|^2) / 2 for the balanced complete r-partite graph
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
    # exhaustive max |E| over all K_{r+1}-free graphs on n labelled vertices
    pairs = [frozenset(p) for p in combinations(range(n), 2)]
    best = -1
    for bits in product((0, 1), repeat=len(pairs)):
        adj = {e for e, b in zip(pairs, bits) if b}
        if not has_clique(adj, r + 1):
            best = max(best, len(adj))
    return best

def clone(adj, n, keep, drop):
    # delete 'drop', re-attach it as a twin of 'keep' (same neighbourhood, not adjacent to keep)
    keep_nbrs = {u for u in range(n) if u != keep and frozenset((u, keep)) in adj}
    new = {e for e in adj if drop not in e}
    for u in keep_nbrs:
        if u != drop:
            new.add(frozenset((u, drop)))
    return new

if __name__ == "__main__":
    # the candidate count is the true maximum, on every small case
    for n in range(2, 8):
        for r in range(1, n):
            assert turan_edges(n, r) == brute_force_max_edges(n, r)

    # one clone step on a guilty triple strictly adds edges, stays K_{r+1}-free
    # path 0-1-2-3, r=2 (forbid K_3); 0,2 non-adjacent, deg(2)=2 > deg(0)=1, clone 2 over 0
    n, r = 4, 2
    G = {frozenset((0, 1)), frozenset((1, 2)), frozenset((2, 3))}
    H = clone(G, n, keep=2, drop=0)
    assert not has_clique(H, r + 1) and len(H) > len(G)
```
