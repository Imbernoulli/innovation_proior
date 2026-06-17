# Centroid decomposition for the exact-length tree path with fewest edges

## Problem

Tree on $n$ vertices, each edge a non-negative integer length, plus a target
$K$. Among all simple paths between two distinct vertices whose lengths sum to
exactly $K$, return the minimum number of edges, or $-1$ if no path totals $K$.

## Key idea

**Divide and conquer over paths-through-a-vertex.** Every path either passes
through a chosen vertex $u$ or lies entirely in one of the pieces left when $u$
is deleted. Handle all paths through $u$ now, delete $u$, and recurse on each
piece. A fixed path is processed at the first deleted vertex on that path; after
that deletion its endpoints are no longer together in any recursive piece, so it
is not processed again.

**Choose $u$ to be the centroid** — the vertex whose deletion leaves every piece
with at most $n/2$ vertices. Such a vertex always exists (at most one piece can
exceed $n/2$; step toward an oversized piece and the walk never backtracks, so
it halts at a balanced vertex), so the recursion has depth $O(\log n)$. The
centroid is found in $O(n)$: root the piece, compute subtree sizes $\mathrm{sz}$,
and pick the vertex minimizing
$\max\!\big(\,\text{(size }-\mathrm{sz}(u)),\ \max_{w \text{ child}} \mathrm{sz}(w)\big)$.

**Combine paths through the centroid in $O(\text{size})$.** A path through $u$ is
two half-paths into *different* neighbor-subtrees of $u$ (or $u$ itself as an
endpoint). DFS each neighbor-subtree to list every $(\text{cost},\text{depth})$ =
(summed length, edge count) from $u$, pruning once cost $> K$. Keep an array
$A[c]$ = minimum depth of a half-path of length $c$ seen in *earlier* subtrees.
Per subtree, **query then fill**: for each $(c,d)$, if $A[K-c]$ is set the glued
path has length exactly $K$ with $d + A[K-c]$ edges (and if $c=K$, depth $d$
alone, with $u$ an endpoint); then record $A[c]\leftarrow\min(A[c],d)$. Querying
before filling guarantees the two halves lie in different subtrees, so no path
doubles back through one edge.

**$O(1)$ reset via a freshness stamp.** $A$ is shared across all centroids over
the whole recursion; clearing $K+1$ slots per centroid would be $O(nK)$. Instead
a per-centroid stamp marks which slots are live: $A[c]$ counts only when
`seen[c]` equals the current stamp, and bumping the stamp invalidates everything
in $O(1)$.

## Complexity

$T(n) = O(n) + \sum_i T(n_i)$ with each $n_i \le n/2$, so $O(\log n)$ levels; each
level does $O(\text{level size})$ work and the level sizes sum to $\le n$, giving
**$O(n\log n)$ time**. Memory is $O(n)$ for the tree plus $O(K)$ for the two
helper arrays.

## Code

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
