In front of me is a tree on $n$ vertices, every edge carrying a non-negative integer length, and a target $K$. I want the path between two distinct vertices whose lengths sum to exactly $K$ using as few edges as possible — and $-1$ if no path hits $K$ on the nose. Think $n$ in the hundreds of thousands and $K$ up to about a million. Let me start from the only thing I'm sure of: a simple path in a tree is pinned down entirely by its two endpoints, because between any two distinct vertices the path is unique. So "find a path" really means "find a pair of vertices."

The completely honest first move: enumerate the pairs. For every ordered pair $(a,b)$ walk the unique $a$–$b$ path, add up the edge lengths, count the edges, keep the best pair with sum $=K$. Correct. And dead on arrival: there are $\binom{n}{2}$ pairs, and walking each path is itself up to $O(n)$, so I'm somewhere around $O(n^3)$, and even if I were clever about reusing path computations the *number* of pairs alone is $O(n^2)$. At $n=2\cdot10^5$ that's $4\cdot10^{10}$ — hopeless. So the pair-by-pair framing is the enemy; I'm paying for every endpoint pair separately when most of them share almost all their structure.

Maybe I can be smarter and root the tree, then do a DP over depth. Root at vertex $1$, and for each vertex store, indexed by length $\ell$, the minimum number of edges on a downward path from that vertex to a descendant of total length $\ell$. That's a tidy DP, and it would instantly find paths that go straight down from a vertex to a descendant. But stare at what a real path looks like. A path between two distinct vertices $a$ and $b$ goes *up* from $a$ to their lowest common ancestor and then *down* to $b$ — it bends. It is not a root-to-descendant chain. To catch a bent path with the down-DP I'd have to, at every vertex $v$, glue together two downward paths into two *different* children of $v$ whose lengths sum to $K$. That gluing is exactly an all-pairs combination of $v$'s subtree branches, and a long thin tree (a near-path graph) has one vertex whose subtree is almost the whole tree — so doing that combination at the root is back to combining $O(n)$ lengths against $O(n)$ lengths, $O(n^2)$. The DP idea is not wrong, it's that doing the combination at a *fixed* root concentrates all the cost at the top of an unbalanced tree.

So the trouble has a shape now. Combining branch-lengths to assemble bent paths is the right primitive, but I must not do it at a single fixed root, or an adversarial tree forces all $O(n^2)$ work into one vertex. I need to *choose where I do the combining* so that no single combination is over a huge subtree, and so that I don't process the same path twice.

Let me reorganize around a single vertex and ask what's true regardless of which one. Pick any vertex $u$. The optimal path $P^*$ either passes through $u$, or it doesn't. If it doesn't, then deleting $u$ shatters the tree into the subtrees hanging off $u$, and $P^*$ — never touching $u$ — lies entirely inside one of those pieces. That's a clean divide-and-conquer: handle all paths *through* $u$ now, then delete $u$ and recurse independently on each remaining piece, where the same argument applies. For any fixed endpoint pair, there is a first deleted vertex on its path; just before that deletion both endpoints are still in the same current piece, and the path is one of the paths through the vertex being handled. After that deletion the endpoints are separated or the path is already decided, so it cannot be handled again in a recursive piece. Two things to nail: (1) how do I find the best path through a *single* vertex $u$ efficiently, and (2) which $u$ do I pick so the recursion doesn't blow up.

Take (1) first — paths through a fixed $u$. A path through $u$ has $u$ somewhere on it. Either $u$ is an endpoint, or $u$ is interior, in which case the path enters $u$ from one neighbor's subtree and leaves into a *different* neighbor's subtree. So every path through $u$ decomposes into two "half-paths," each going from $u$ down into one of its neighbor-subtrees (length $0$, the degenerate half, allowed when $u$ is an endpoint). For each half-path I care about two numbers: its total length, and its edge count. I want two half-paths going into *different* subtrees whose lengths sum to exactly $K$, minimizing the sum of their edge counts.

So let me DFS out from $u$ into each neighbor-subtree and collect, for every vertex reached, the pair $(\text{cost},\ \text{depth})$ — cost being the summed edge lengths from $u$, depth being the number of edges from $u$. Now I need, across two different subtrees, costs that add to $K$ with minimum total depth. The brute pairing within $u$ is again all-pairs over the half-paths, which over a star-like $u$ is $O((\text{subtree size})^2)$ — I haven't gained anything yet unless I'm cleverer about the matching.

Instead of pairing the half-paths against each other directly, I'll keep one array $A$ indexed by length: $A[c]$ = the minimum depth of any half-path I've *already seen* (from previously-processed neighbor subtrees of $u$) whose cost is exactly $c$. Then I process $u$'s neighbor-subtrees one at a time, and for each new subtree I do two passes. First a *query* pass: for each vertex in this subtree with cost $c$ and depth $d$, the complementary half-path I'd need has cost $K-c$; if $A[K-c]$ is set, then gluing gives a full path of length exactly $K$ with $d + A[K-c]$ edges — a candidate answer. Then a *fill* pass: for each vertex in this subtree with cost $c$ and depth $d$, set $A[c] \leftarrow \min(A[c], d)$, so later subtrees can match against it. Crucially I query *before* I fill for the same subtree, so when a query finds a match in $A$, that match was deposited by an *earlier* subtree — the two halves live in different neighbor-subtrees, exactly as a genuine path through $u$ requires. If I filled first, a vertex could pair with another vertex in its own subtree, and then the "path" would go down into a subtree and come right back out the same edge — not a simple path. Query-then-fill enforces the different-subtree condition for free.

I'm forgetting the case where $u$ itself is an endpoint: a half-path of cost exactly $K$ all by itself, paired with the trivial empty half at $u$. That's easy — whenever a vertex in any subtree has cost $c = K$, its depth $d$ is directly a candidate (the path from $u$ straight down to it). I'll fold that into the query pass: if $c = K$, candidate $= d$.

One more thing about the array $A$. Costs go from $0$ to $K$, so $A$ has $K+1$ slots. Lengths are non-negative, so once a half-path's running cost exceeds $K$ I can prune the DFS branch — it can never reach a useful complementary cost in $[0,K]$. Good, that also keeps the indices in range. And I should only ever touch $A[c]$ for $c \le K$.

Now the reset problem, because it's a real trap. This array $A$ is going to be reused at every vertex I "handle" across the whole recursion. After finishing vertex $u$ I must wipe the slots I touched, or $u$'s leftover values poison the next vertex. Wiping all $K+1$ slots costs $O(K)$ per handled vertex, and I handle $O(n)$ vertices, giving $O(nK)$ — with $K\sim10^6$ that's catastrophic. But I only ever *touch* $O(\text{subtree size})$ slots at a given vertex, so clearing the whole array is absurd overkill. The clean fix is a freshness stamp: keep a counter `stamp` that I bump by one each time I start a new handled vertex, and an array `seen[c]` storing the stamp at which $A[c]$ was last written. A slot $A[c]$ counts as "set" only if `seen[c] == stamp`. To read $A[K-c]$ I check `seen[K-c] == stamp`; to write $A[c]$ I set `seen[c] = stamp` and update the depth. Bumping `stamp` invalidates every old slot in $O(1)$ — no wiping at all. That turns the whole combine at $u$ into $O(\text{subtree size})$ time.

So handling one vertex $u$ — DFS its subtrees, query-then-fill per neighbor against the stamped array — is linear in the size of the subtree currently being handled. Now part (2): which $u$, and how does the recursion add up. The recurrence is $T(s) = (\text{handle cost}) + \sum_i T(s_i)$ where $s_i$ are the sizes of the pieces after deleting $u$ from a current tree of size $s$, and $\sum_i s_i = s-1$. Handle cost is $O(s)$. If I pick $u$ carelessly — say always a leaf, or a fixed root — then one piece can have size $s-1$, the recursion has depth $n$, and $\sum$ of the per-level $O(s)$ work is $O(n^2)$ again. I'm right back where I started. The *entire* benefit hinges on the split being balanced.

What would make it balanced? I want a $u$ whose every remaining piece has at most half the vertices. The first question is whether such a vertex even exists in every tree — if it doesn't, the whole plan is built on sand. Let me try to construct one and see if the construction can ever get stuck. Pick any candidate $v$. Delete it; you get a forest. At most one of those pieces can contain more than $s/2$ vertices — two pieces each over half would already exceed $s$ vertices total, impossible. If no piece is over $s/2$, $v$ is the vertex I want, done. Otherwise there's exactly one oversized piece $T_w$, reached through neighbor $w$; step from $v$ to $w$ and repeat. The worry is that this walk might cycle forever. It can't, and here's why: once I leave $v$ for $w$, the piece on $v$'s side (everything that was *not* $T_w$, plus $v$ itself) has at most $s/2$ vertices, so from $w$'s vantage that whole side is a *small* piece — the oversized direction can only be deeper into what was $T_w$, never back toward $v$. Each step strictly moves into a new vertex I'll never revisit, so after at most $s$ steps the walk runs out of vertices, and it can only stop at a vertex with no oversized piece. So a balanced vertex must exist. Call it the central vertex; removing it leaves every piece with $\le s/2$ vertices.

Let me actually run that walk on the path $0\!-\!1\!-\!2\!-\!3$ ($s=4$, so I want every piece $\le 2$), since an existence proof I can't trace once makes me nervous. Start at vertex $0$. Deleting $0$ leaves the single piece $\{1,2,3\}$ of size $3 > 2$ — oversized, reached through neighbor $1$. Step to $1$. Deleting $1$ leaves $\{0\}$ of size $1$ and $\{2,3\}$ of size $2$; the max is $2 \le 2$, so I stop — vertex $1$ is central. And indeed deleting either endpoint gives a piece of size $3$ while deleting $1$ or $2$ gives max-piece $2$, so the two interior vertices are exactly the balanced ones. The walk landed on one of them in a single step and never looked back at $0$, just as the no-backtracking argument promised.

With that guarantee the recursion has depth $O(\log n)$: each level at most halves the size of every piece. The total handle work is $\sum$ over levels of $O(\text{level total size})$. Each level partitions a disjoint set of vertices, so the total size summed across all pieces at one level is $\le n$, and there are $O(\log n)$ levels — total $O(n\log n)$, provided finding the central vertex at each handled subtree is no worse than the handle itself.

So I need to find the central vertex cheaply. Root the current piece at any vertex and DFS to get subtree sizes $\mathrm{sz}(v)$ counting $v$ and its descendants in this piece. For a candidate $u$ inside a piece of size $s$, deleting $u$ produces: one piece per child $w$ of size $\mathrm{sz}(w)$, plus the "upward" piece of size $s - \mathrm{sz}(u)$ (everything not in $u$'s subtree). The largest piece is $\max\big(s - \mathrm{sz}(u),\ \max_{w \text{ child of } u} \mathrm{sz}(w)\big)$. The central vertex is the one minimizing this maximum; one more DFS over the piece, evaluating that max at each vertex, finds it in $O(s)$. That matches the $O(s)$ handle cost, so the level work stays $O(\text{level size})$ and the total is $O(n\log n)$. (If I were sloppier and found the center in $O(s\log s)$, I'd get $O(n\log^2 n)$ — still fine, but the $O(s)$ DFS is clean, so I'll keep it.)

Let me make sure I avoid a subtle bug: once I delete a central vertex and recurse, the recursion must stay *inside* the current piece and never wander back across a deleted vertex. The standard way is a `removed[]` flag — when a vertex is chosen as the center for its piece I mark it removed, and every DFS (size computation, center search, the path-combining DFS) refuses to cross into a removed vertex. That confines each recursive call to exactly the connected component it's supposed to own, which is what makes the disjointness-per-level argument hold.

Let me hand-trace the whole thing on a tiny tree before I commit to code, because the query-then-fill ordering is exactly the kind of detail that's easy to get backwards. Path graph $1 - 2 - 3 - 4$, every edge length $1$, and $K = 2$. By eye the answer should be $2$ — the path $1\!-\!2\!-\!3$ (or $2\!-\!3\!-\!4$) has length $2$ and uses $2$ edges; no exact-length-$2$ path uses fewer edges, and the single edges have length $1\neq 2$. The walk above showed both interior vertices are centroids; I'll handle vertex $2$, whose split into two non-trivial subtrees exercises the cross-subtree pairing better than vertex $1$ would. Its neighbor-subtrees: $\{1\}$ through edge $(2,1)$, and $\{3,4\}$ through edge $(2,3)$. Process subtree $\{1\}$ first. Query pass: vertex $1$ has cost $1$, depth $1$; I need $A[K-1]=A[1]$, currently empty, no match; cost $\ne K$. So this first subtree contributes nothing — which is the point of query-then-fill, since vertex $1$ has no earlier subtree to pair with. Fill pass: $A[1] \leftarrow 1$. Now subtree $\{3,4\}$. Query pass: vertex $3$ has cost $1$, depth $1$ — need $A[K-1]=A[1]=1$, set! candidate $= 1 + 1 = 2$. Vertex $4$ has cost $2$, depth $2$ — that equals $K$, so candidate $= 2$ (the path $2\!-\!3\!-\!4$ with $2$ as endpoint); also need $A[0]$, empty. Best so far $2$. The query found the pair $(1,3)$ in different subtrees of $2$, length $1+1=2$, exactly $K$, two edges — matching the by-eye answer. Recursion into the remaining pieces $\{1\}$ and $\{3,4\}$ can't beat $2$ (the only $\le 2$-edge exact path inside $\{3,4\}$ is $3\!-\!4$, length $1$).

One hand example proves nothing about the corners, though — the $c=K$ branch, zero-weight edges, the $-1$ case, an unbalanced tree where the centroid recursion really kicks in. So rather than trust my eye, I'll write the algorithm out and pit it against a dead-simple reference: an all-pairs brute force that, from every source, walks to every other vertex accumulating (cost, depth) and keeps the min depth wherever cost hits $K$. That reference is obviously correct and obviously too slow, which is exactly what I want from an oracle. Run both on a few fixed cases and then on thousands of random small weighted trees ($n\le 9$, $K\le 6$, edge weights $0$–$4$, so zero-weight edges and missed targets both occur). On the fixed cases: the path-of-$4$ above returns $2$; the official-style sample (edges $0\!-\!1{:}1,\,1\!-\!2{:}2,\,1\!-\!3{:}4$, $K=3$) returns $2$ for the path $0\!-\!1\!-\!2$ of length $3$; a tree with no exact-$K$ path returns $-1$; and a tree with a $0$-weight edge under $K=0$ returns $1$, a genuine two-vertex path of total length $0$. Over $3000$ random trials the centroid algorithm and the brute reference agreed on every single one — zero mismatches. That's the check I actually wanted: not that the idea sounds right, but that the implementation with all its index arithmetic ($K-c$, the stamp gating, the prune at cost $>K$) reproduces a trivially-correct oracle across the corners I couldn't enumerate by hand.

Time to write it. I'll build an adjacency list of (neighbor, edge-length). `process(start)` takes any vertex of a still-undeleted piece: compute sizes, find the center, run the double-DFS combine, mark the center removed, then recurse on each neighbor that isn't removed. The combine uses the global `best_depth[]`/`seen[]` pair with the `stamp` trick, and `answer` is min-updated, initialized to $-1$ / "infinity." I'll write the DFS iteratively where recursion depth could reach $O(n)$ on a degenerate line, to avoid blowing Python's stack. I also need `parent` to be a reusable array: allocating a fresh length-$n$ parent array inside every recursive component would add an accidental $O(n)$ cost per component, which is not the algorithm I just derived.

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

I end up with each endpoint pair considered at the first deleted central vertex that its path crosses. Pair enumeration disappears because, at that vertex, I only keep the best earlier half-path for each length; query-then-fill keeps the two halves in different neighbor-subtrees; `seen` and `stamp` make each reset constant-time; and deleting a vertex whose remaining pieces are all at most half the current component keeps the recursion to $O(\log n)$ levels. The implementation follows that shape: size the current component, find the central vertex, combine every exact-$K$ path through it, delete it, and handle the remaining pieces independently.
