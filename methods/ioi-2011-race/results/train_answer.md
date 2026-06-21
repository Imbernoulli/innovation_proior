We are given a tree on $n$ vertices, each edge carrying a non-negative integer length, and a target $K$. Among all simple paths between two distinct vertices whose edge lengths sum to exactly $K$, I want the one with the fewest edges — and $-1$ if no path totals $K$. With $n$ in the hundreds of thousands and $K$ up to about a million, the structure of the problem is set by one fact: a simple path in a tree is pinned down entirely by its two endpoints, since the path between any two distinct vertices is unique. So finding a path means finding a pair of vertices. The honest first approach — enumerate every pair $(a,b)$, walk the unique $a$–$b$ path, sum its lengths, and keep the best pair summing to $K$ — is correct but hopeless: the number of pairs alone is $\binom{n}{2}=O(n^2)$, and walking each path adds another factor of $n$. At $n=2\cdot10^5$ that is already $4\cdot10^{10}$ pairs, before any path-walking. The framing itself is the enemy, because it pays for each endpoint pair separately even though neighboring pairs share almost all their structure.

The natural repair — root the tree and run a DP over depth, storing for each vertex the minimum edge count of a downward path of each total length — catches root-to-descendant paths instantly but misses what a real path looks like. A path between $a$ and $b$ goes *up* to their lowest common ancestor and then *down*; it bends. To catch a bent path with the down-DP I would have to glue, at every vertex, two downward half-paths into two *different* children whose lengths sum to $K$, and that gluing is an all-pairs combination over the vertex's subtree branches. On a long thin near-path tree one vertex owns almost the entire tree, so the combination at that vertex is again $O(n^2)$. The DP idea is not wrong; the failure is that doing the combination at a *fixed* root concentrates all the cost at the top of an unbalanced tree. The lesson is sharp: combining branch-lengths to assemble bent paths is the right primitive, but I must choose *where* I do the combining so that no single combination runs over a huge subtree, and so that no path is processed twice.

The method is centroid decomposition. Pick a vertex $u$; the optimal path either passes through $u$ or, if it does not, lies entirely inside one of the pieces that remain when $u$ is deleted, because deleting $u$ shatters the tree into the subtrees hanging off it and a path that never touches $u$ stays inside one piece. So I handle every exact-$K$ path *through* $u$ now, delete $u$, and recurse independently on each remaining piece, where the same dichotomy applies. For any fixed endpoint pair there is a first deleted vertex on its path; just before that deletion both endpoints sit in the same current piece and the path is one of the paths through the vertex being handled, and after it the endpoints are separated, so the path is handled exactly once. Two things must be made precise: how to find the best path through a single vertex efficiently, and which vertex to pick so the recursion does not blow up.

For the path-through-$u$ subproblem, observe that a path through $u$ is two "half-paths," each running from $u$ down into one of its neighbor-subtrees — and the degenerate empty half is allowed, which is the case where $u$ itself is an endpoint. For each half-path I track its summed edge length (its *cost*) and its edge count (its *depth*), and I want two half-paths into *different* subtrees whose costs sum to exactly $K$ with minimum total depth. Rather than pairing half-paths against each other directly — which is again all-pairs — I keep one array indexed by length, $A[c]$ = the minimum depth of any half-path of cost exactly $c$ already seen from previously-processed neighbor-subtrees of $u$. I process $u$'s neighbor-subtrees one at a time, and for each I do two passes. First a *query* pass: for each vertex of cost $c$ and depth $d$, the complement I need has cost $K-c$, so if $A[K-c]$ is live then gluing yields a full path of length exactly $K$ with $d+A[K-c]$ edges — a candidate. Then a *fill* pass: for each vertex of cost $c$, depth $d$, set $A[c]\leftarrow\min(A[c],d)$ so later subtrees can match against it. Querying *before* filling for the same subtree is the load-bearing choice: when a query finds a match in $A$, that match was deposited by an *earlier* subtree, so the two halves live in different neighbor-subtrees, exactly as a genuine simple path through $u$ demands. Had I filled first, a vertex could pair with another in its own subtree and the "path" would descend into a subtree and return up the same edge — not simple. Query-then-fill enforces the different-subtree condition for free. The endpoint case folds into the query pass: whenever a vertex has cost $c=K$, its depth $d$ is directly a candidate (the path straight down from $u$). Costs run from $0$ to $K$, so $A$ has $K+1$ slots, and since lengths are non-negative I prune any DFS branch the moment its running cost exceeds $K$ — it can never reach a useful complement and would index out of range.

The reset is a real trap. The array $A$ is reused at every handled vertex across the whole recursion, and after finishing $u$ its leftover values must not poison the next vertex. Wiping all $K+1$ slots costs $O(K)$ per handled vertex over $O(n)$ vertices — $O(nK)$, catastrophic at $K\sim10^6$ — yet I only ever *touch* $O(\text{subtree size})$ slots at a vertex, so clearing the whole array is absurd overkill. The fix is a freshness stamp: a counter `stamp` bumped by one each time a new vertex is handled, and an array `seen[c]` holding the stamp at which $A[c]$ was last written. Slot $A[c]$ is live only when `seen[c] == stamp`; reading $A[K-c]$ checks `seen[K-c] == stamp`, writing $A[c]$ sets `seen[c] = stamp`. Bumping `stamp` invalidates every old slot in $O(1)$ with no wiping, so the combine at $u$ is $O(\text{subtree size})$.

What makes the whole thing fast is the choice of which vertex to delete. The recurrence is $T(s)=O(s)+\sum_i T(s_i)$ with $\sum_i s_i=s-1$. If I pick carelessly — a leaf, or a fixed root — one piece can have size $s-1$, the recursion has depth $n$, and the summed per-level $O(s)$ work is $O(n^2)$ again. The entire benefit hinges on a balanced split, so I delete the *centroid*: the vertex whose removal leaves every remaining piece with at most $s/2$ vertices. Such a vertex always exists. Delete any candidate $v$; at most one resulting piece can exceed $s/2$, since two such pieces would already exceed $s$ vertices. If none does, $v$ is the centroid. Otherwise step from $v$ into the single oversized piece through its neighbor $w$ and repeat — and this walk never backtracks, because once I leave $v$ for $w$ the entire side that was *not* the oversized piece (plus $v$) has at most $s/2$ vertices, so from $w$ the oversized direction can only lie deeper into what was the oversized piece, never back toward $v$. Each step moves into a fresh vertex, so the walk halts within $s$ steps, and it can only halt where no piece is oversized — the centroid. With it, every piece has $\le s/2$ vertices, the recursion has depth $O(\log n)$, each level partitions a disjoint vertex set so the level sizes sum to $\le n$, and the total handle work is $O(n\log n)$.

Finding the centroid must itself be cheap. Root the current piece anywhere and DFS to get subtree sizes $\mathrm{sz}(v)$. Deleting a candidate $u$ in a piece of size $s$ produces one piece per child $w$ of size $\mathrm{sz}(w)$ plus the "upward" piece of size $s-\mathrm{sz}(u)$; the largest piece is $\max\!\big(s-\mathrm{sz}(u),\ \max_{w\text{ child of }u}\mathrm{sz}(w)\big)$, and the centroid minimizes this maximum. One more DFS over the piece evaluates it at every vertex in $O(s)$, matching the $O(s)$ handle cost. To keep each recursive call confined to exactly its own connected component — never wandering back across a deleted vertex — a `removed[]` flag marks chosen centroids, and every DFS (sizing, centroid search, half-path collection) refuses to cross into a removed vertex; this is what makes the disjoint-per-level argument hold. One sanity check on the tiny path $1\!-\!2\!-\!3\!-\!4$ with unit edges and $K=2$: handling centroid $2$, processing subtree $\{1\}$ first fills $A[1]\leftarrow1$, then in subtree $\{3,4\}$ vertex $3$ (cost $1$, depth $1$) queries $A[K-1]=A[1]=1$ and yields $1+1=2$, while vertex $4$ (cost $2=K$, depth $2$) yields $2$ directly — answer $2$, as it should be.

So in the implementation, `process(start)` takes any vertex of a still-undeleted piece, computes subtree sizes, finds the centroid, runs the query-then-fill double-DFS combine over its neighbor-subtrees against the stamped `best_depth`/`seen` arrays, min-updates `answer`, marks the centroid removed, and recurses on each remaining piece. The DFS is iterative so a degenerate line graph cannot overflow Python's stack, and `parent` is a reusable length-$n$ array rather than one allocated per component, which would silently add $O(n)$ work per component and break the very complexity I derived.

```python
import sys

def race(n, K, edges):
    """edges: list of (u, v, w) with 0-based vertices and non-negative w.
    Returns the minimum number of edges on a path of total length exactly K,
    or -1 if no such path exists. O(n log n)."""
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))

    removed = [False] * n
    size = [0] * n
    parent = [-1] * n

    # best_depth[c] = min #edges of a centroid->node half-path of total length
    # c, among neighbor-subtrees processed SO FAR. Live only when seen[c] equals
    # the current stamp (a per-centroid stamp gives O(1) reset, no O(K) wipe).
    best_depth = [0] * (K + 1)
    seen = [-1] * (K + 1)
    stamp = 0
    answer = -1

    def calc_size(root):
        order = []
        st = [root]
        parent[root] = root
        while st:                          # iterative DFS over the component
            cur = st.pop()
            order.append(cur)
            for nxt, _ in adj[cur]:
                if not removed[nxt] and nxt != parent[cur]:
                    parent[nxt] = cur
                    st.append(nxt)
        for cur in order:
            size[cur] = 1
        for cur in reversed(order):        # children before parents
            if parent[cur] != cur:
                size[parent[cur]] += size[cur]
        return order, parent

    def find_centroid(order, parent, total):
        best, best_max = order[0], total + 1
        for cur in order:
            mx = total - size[cur]         # the "upward" piece
            for nxt, _ in adj[cur]:
                if not removed[nxt] and nxt != parent[cur]:
                    if size[nxt] > mx:
                        mx = size[nxt]     # a child subtree
            if mx < best_max:
                best_max, best = mx, cur
        return best

    def dfs_collect(start, c0, centroid):
        # (cost, depth) of every half-path into the subtree entered via 'start';
        # 'centroid' is start's parent, so the DFS never crosses back through it
        # into a sibling subtree. prune once cost > K.
        out = []
        st = [(start, c0, 1, centroid)]
        while st:
            cur, cost, depth, par = st.pop()
            if cost > K:
                continue
            out.append((cost, depth))
            for nxt, w in adj[cur]:
                if not removed[nxt] and nxt != par:
                    st.append((nxt, cost + w, depth + 1, cur))
        return out

    def process(start):
        nonlocal stamp, answer
        order, parent = calc_size(start)
        if len(order) == 1:
            return
        total = size[start]
        c = find_centroid(order, parent, total)

        stamp += 1
        for nb, w in adj[c]:               # one neighbor-subtree at a time
            if removed[nb]:
                continue
            half = dfs_collect(nb, w, c)
            for cost, depth in half:       # query against EARLIER subtrees
                if cost == K and (answer == -1 or depth < answer):
                    answer = depth         # centroid is an endpoint
                need = K - cost
                if 0 <= need <= K and seen[need] == stamp:
                    cand = depth + best_depth[need]
                    if answer == -1 or cand < answer:
                        answer = cand
            for cost, depth in half:       # then fill, visible to LATER subtrees
                if seen[cost] != stamp or depth < best_depth[cost]:
                    seen[cost] = stamp
                    best_depth[cost] = depth

        removed[c] = True                  # delete centroid, recurse on pieces
        for nb, _ in adj[c]:
            if not removed[nb]:
                process(nb)

    sys.setrecursionlimit(1 << 20)
    process(0)
    return answer


if __name__ == "__main__":
    data = sys.stdin.buffer.read().split()
    if data:
        it = iter(data)
        n = int(next(it)); K = int(next(it))
        edges = [(int(next(it)), int(next(it)), int(next(it)))
                 for _ in range(n - 1)]
        print(race(n, K, edges))
```
