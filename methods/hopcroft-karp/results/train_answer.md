We are given a bipartite graph with left part $L$ of size $n$, right part $R$ of size $m$, and edges only across the split, supplied as a $0$-based adjacency list. We want a maximum-cardinality matching: the largest set of edges with no two sharing an endpoint, reported as the matching size together with the matched $(u,v)$ pairs. Enumerating edge subsets is hopeless, but the natural local state of a matching is already informative: for each left vertex its current right partner in `match_l`, and for each right vertex its current left partner in `match_r`. The real question is how to grow a valid matching into a larger one without ever violating disjointness. The answer is the augmenting path. If I take a path that alternates between edges outside the current matching and edges inside it, starting and ending at free vertices, then because the first and last edges are unmatched the path holds one more unmatched edge than matched edge; flipping membership along it deletes the matched edges, inserts the unmatched ones, leaves every internal vertex with exactly one incident matched edge, and turns both free endpoints into matched ones, so the matching size goes up by one. Berge's criterion makes this complete: if a matching $M$ has no augmenting path yet some larger $M^*$ exists, then in the symmetric difference $M \oplus M^*$ every vertex has degree at most two, so the components are alternating cycles and paths; cycles split their edges evenly between the two matchings, and since $M^*$ is larger, some path component carries more $M^*$ edges than $M$ edges, which forces it to start and end with $M^*$ edges and hence to be an augmenting path for $M$ — a contradiction. So a matching is maximum exactly when no augmenting path remains, and this also gives the stopping test.

The naive realization — find one augmenting path by a single oriented DFS from a free left vertex, flip it, and repeat — is correct but not organized enough. In a bipartite graph the search orients cleanly (unmatched edges go left to right, a matched right vertex continues only through its matched edge back to the left), but each improvement can cost a full graph scan and there can be a linear number of improvements, leaving the worst case too slow. The waste is that one DFS commits to a single path while many independent improvements of the same length may sit available simultaneously.

I propose Hopcroft-Karp, which removes that waste by batching the search. The idea is to measure the alternating distance from all free left vertices at once and then, in a single coordinated pass, augment along a whole maximal set of vertex-disjoint shortest augmenting paths. Each round is a phase with two stages. The first stage, `build_layers`, is a breadth-first sweep that assigns left-side layers: every free left vertex gets $\text{level}=0$ and seeds the queue; expanding a left vertex $u$ at $\text{level}[u]=k$, each edge $u\!-\!v$ either reaches a free right vertex — recording the first such distance as `free_level`, the count of unmatched edges $k+1$ in a shortest augmenting path of total length $2(k+1)-1$ — or reaches a matched right vertex whose partner $w=\text{match\_r}[v]$ becomes a left vertex of the next layer at $\text{level}[w]=k+1$. I store only left-side levels because the right side is determined by the matching. Crucially, once the step $\text{next\_level}=\text{level}[u]+1$ would reach or exceed `free_level`, expansion stops there: going farther could only build augmenting paths longer than the current shortest, which this phase must not touch. The second stage, `find_path`, is a DFS confined to that layered graph. From left vertex $u$, an edge to a free right vertex is accepted only when $\text{level}[u]+1=\text{free\_level}$, and an edge to a matched right vertex $v$ is followed only when $\text{level}[u]+1<\text{free\_level}$ and the partner $w=\text{match\_r}[v]$ sits exactly one layer deeper, $\text{level}[w]=\text{level}[u]+1$; a successful recursion flips the edges as it unwinds. The single trick that makes the batch vertex-disjoint and the phase linear is marking a left vertex dead by setting $\text{level}[u]=\text{INF}$ on the way out — both when it fails (all of its in-layer routes are exhausted, so never try it again this phase) and when it succeeds (it has just been consumed by one of the chosen paths). Because every accepted path lands on the first free-right layer, every recursive step advances exactly one layer, and consumed vertices are pulled out of later searches, the phase produces a maximal vertex-disjoint set of shortest augmenting paths.

What makes the algorithm fast is that this phase structure forces the shortest augmenting path to grow. Let a phase pick a maximal set $A=\{P_1,\dots,P_t\}$ of vertex-disjoint paths of edge length $\ell$, and let $M'=M\oplus(P_1\cup\cdots\cup P_t)$, of size $|M|+t$. Suppose an augmenting path $P$ for $M'$ remains. If $P$ is vertex-disjoint from every $P_i$, then no edge on $P$ changed status, so $P$ was already augmenting for $M$; maximality of $A$ forbids $|P|=\ell$ and the definition of $\ell$ forbids anything shorter. The only live case is that $P$ touches some $P_i$. At a shared vertex, the endpoints of $P$ are free under $M'$ while every vertex of the chosen paths is matched under $M'$, so the shared vertex is internal to $P$, and the unique $M'$-matched edge incident there must lie on both $P$ and that $P_i$ (a matching offers only one). Hence $P$ shares at least one edge with $\bigcup A$, so $(P_1\cup\cdots\cup P_t)\oplus P$ has at most $t\ell+|P|-1$ edges. But $M'\oplus P$ has size $|M|+t+1$, so this same symmetric difference $M\oplus(M'\oplus P)$ must decompose into at least $t+1$ augmenting paths for $M$, each of length at least $\ell$, giving at least $(t+1)\ell$ edges. Therefore

$$t\ell + |P| - 1 \;\ge\; (t+1)\ell \quad\Longrightarrow\quad |P| \ge \ell + 1,$$

and since augmenting paths in a bipartite graph have odd length, the next achievable length is at least $\ell+2$. The shortest augmenting path strictly lengthens every phase.

That alone does not bound the phase count, so I split the run by length. Let $s$ be the maximum matching size and set $p=\max(1,\lfloor\sqrt{s}\rfloor)$. After $p$ phases the shortest remaining augmenting path, if any, has length at least $2p+1$. Comparing the current $M$ against a maximum $M^*$, the symmetric difference contains exactly $s-|M|$ vertex-disjoint augmenting paths, each now of length at least $2p+1$ and therefore carrying at least $p$ current-matching edges; those edges are disjoint across paths and number at most $s$, so $s-|M|\le s/p=O(\sqrt s)$. The first $p$ phases cost $O(\sqrt s)$ and only $O(\sqrt s)$ augmentations remain afterward, each consuming at least one more phase, so with $s\le (n+m)/2$ the total is $O(\sqrt V)$ phases for $V=n+m$. Each phase scans every edge once in the BFS layering and once in the layered DFS, with dead vertices preventing repeated failed exploration, giving $O(E\sqrt V)$ time and $O(V+E)$ space. The loop ends precisely when `build_layers` cannot reach any free right vertex, which by Berge's criterion certifies the matching as maximum.

```python
import sys


def read_bipartite(data):
    """Parse n (left size), m (right size), e (edge count), then e edges
    (u in [0, n), v in [0, m)) into a 0-based adjacency list adj[u] -> [v, ...]."""
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    e = int(next(it))
    adj = [[] for _ in range(n)]
    for _ in range(e):
        u = int(next(it))
        v = int(next(it))
        adj[u].append(v)
    return n, m, adj


def max_matching(n, m, adj):
    """Return maximum matching size and matched (left, right) pairs."""
    sys.setrecursionlimit(max(sys.getrecursionlimit(), n + m + 10))
    match_l = [-1] * n
    match_r = [-1] * m
    INF = n + m + 1
    level = [INF] * n
    free_level = INF

    def build_layers():
        nonlocal free_level
        queue = []
        head = 0
        free_level = INF
        for u in range(n):
            if match_l[u] == -1:
                level[u] = 0
                queue.append(u)
            else:
                level[u] = INF
        while head < len(queue):
            u = queue[head]
            head += 1
            next_level = level[u] + 1
            if next_level >= free_level:
                continue
            for v in adj[u]:
                w = match_r[v]
                if w == -1:
                    free_level = next_level
                elif level[w] == INF:
                    level[w] = next_level
                    queue.append(w)
        return free_level != INF

    def find_path(u):
        next_level = level[u] + 1
        for v in adj[u]:
            w = match_r[v]
            if w == -1:
                if next_level == free_level:
                    match_l[u] = v
                    match_r[v] = u
                    level[u] = INF
                    return True
            elif next_level < free_level and level[w] == next_level and find_path(w):
                match_l[u] = v
                match_r[v] = u
                level[u] = INF
                return True
        level[u] = INF
        return False

    while build_layers():
        for u in range(n):
            if match_l[u] == -1 and level[u] == 0:
                find_path(u)
    pairs = [(u, match_l[u]) for u in range(n) if match_l[u] != -1]
    return len(pairs), pairs


def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    n, m, adj = read_bipartite(data)
    size, pairs = max_matching(n, m, adj)
    out = [str(size)]
    for u, v in pairs:
        out.append(f"{u} {v}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
```
