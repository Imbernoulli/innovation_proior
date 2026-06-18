The object in front of me is an undirected graph on $N$ vertices, numbered $0$ through $N-1$, and a pile of $Q$ queries. Each query hands me four numbers $(S, E, L, R)$ and asks a yes/no question about a walk from $S$ to $E$ that runs in two phases: at the start I may only step on vertices whose number is at least $L$, then at exactly one moment I flip a switch at some vertex, and from that vertex onward I may only step on vertices whose number is at most $R$. The switch vertex itself has to be legal for both halves, so it sits in $[L,R]$. I want to decide each query. Let me start honestly: write the slow thing first, see exactly where it hurts, and let the pain tell me what structure to exploit.

The most literal reading is a two-color BFS. Make a graph with $2N$ states: $(v,0)$ means "at $v$, still in the first phase", $(v,1)$ means "at $v$, already switched". Every original edge $vu$ gives me $(v,0)\!-\!(u,0)$ provided I'm allowed to be on those vertices in phase one (number $\ge L$), and $(v,1)\!-\!(u,1)$ provided I'm allowed in phase two (number $\le R$). At any vertex with number in $[L,R]$ I can take the free switch edge $(v,0)\to(v,1)$. Then the query is just: is $(E,1)$ reachable from $(S,0)$? One BFS, $O(N+M)$ per query. Correct, simple, and it dies the moment $Q$ and $N$ are both large: $O(Q(N+M))$ is a few hundred thousand times a few hundred thousand. I need to stop touching the whole graph once per query.

Before I optimize, let me strip the two-color trick down to its essence, because the $2N$-state graph hides what's really going on. Phase one is a walk from $S$ that never leaves the set of vertices with number $\ge L$. So whatever vertex I'm sitting on when I flip the switch, it has to be reachable from $S$ using only vertices $\ge L$ — that's the connected component of $S$ in the subgraph induced by $\{v : v \ge L\}$. Call that set $V_L$. Phase two is a walk from the switch vertex to $E$ using only vertices $\le R$. The graph is undirected, so "I can get from the switch vertex to $E$ using vertices $\le R$" is the same as "$E$ can get to the switch vertex using vertices $\le R$" — that's the component of $E$ in the subgraph induced by $\{v : v \le R\}$. Call that $V_R$. The switch vertex must live in both $V_L$ and $V_R$; and any vertex in $V_L$ already has number $\ge L$, any vertex in $V_R$ already has number $\le R$, so a vertex in the intersection automatically sits in $[L,R]$ — the explicit switch constraint comes for free. So the entire question collapses to: **do $V_L$ and $V_R$ share a vertex?** That's a much cleaner thing than a two-color BFS.

So a query is an intersection test between two sets. The slow version recomputes both sets from scratch every time — two BFS floods, $O(N+M)$ each. The waste is obvious: across queries, the same $L$ (or a nearby one) keeps producing almost the same $V_L$. I want to compute these reachable sets once, as a structure, and then answer each query by *reading* from the structure instead of flooding.

The lever I haven't used is monotonicity. Look at $V_L$ as $L$ varies. The subgraph "vertices $\ge L$" only grows as $L$ goes down — lowering the threshold never removes a vertex, only adds vertices and the edges among them. So as $L$ decreases from $N-1$ to $0$, components only ever *merge*; they never split. That's an incremental-union process, and I have a tool built precisely for unions: disjoint-set union. Imagine sweeping $L$ from high to low. Start with no vertices present. Lower the threshold one notch at a time; at threshold $\ell$ the vertex $\ell$ "turns on", and I union it with every already-on neighbor (the ones with number $> \ell$, which are exactly the neighbors that turned on earlier). After processing down to threshold $L$, the DSU component containing $S$ is exactly $V_L$. Symmetrically, raising a threshold from $0$ to $R$ and turning vertices on in increasing order, the component of $E$ after reaching $R$ is exactly $V_R$.

This is progress, but a DSU gives me a *current* partition, and I have $Q$ different thresholds; I can't keep one live DSU and ask about all thresholds at once. I could sort queries by $L$ and replay the sweep, recording $V_L$ for each query at the moment its threshold is hit — that's the offline trick. But $V_L$ is an arbitrary subset of vertices; recording it means recording a whole set, and then intersecting two arbitrary sets is back to $O(N)$ per query. The DSU sweep tells me *which component*, but a component is a bag of vertex numbers scattered across $[0,N-1]$ with no structure I can intersect cheaply. I need the components to have a shape I can summarize in $O(1)$ and intersect in $O(1)$.

Let me sit with that wish: I want every component that ever appears during the merge sweep to be describable by a tiny constant-size summary. On a path graph — vertices laid in a line — this would be automatic: a "vertices $\ge L$ reachable from $S$" set on a line is just a contiguous stretch of the line, summarizable by its two endpoints, and two stretches intersect iff their endpoint-intervals overlap. The line is easy precisely because reachable sets are intervals there. The general graph isn't a line, so $V_L$ isn't an interval of the vertex numbers. But maybe I can *relabel* the vertices — assign each vertex a new position on some line — so that every component that the merge sweep ever produces becomes a contiguous interval *in the new positions*. If I could do that, $V_L$ would again be an interval, and I'd be back in the easy world.

When is a family of sets simultaneously interval-izable under one ordering? When the sets form a laminar family — nested or disjoint, never crossing. And the merge sweep produces exactly that: every component at threshold $\ell$ is a union of components from a higher threshold; as $\ell$ drops, components only fuse, so any two components I ever see are either disjoint (different thresholds haven't merged them yet) or one contains the other (a later, lower threshold swallowed the earlier one). Laminar. A laminar family on $N$ leaves is a tree. So the merge history *is* a tree, and I should build it explicitly instead of throwing it away.

Let me build that tree from the high-to-low sweep. The leaves are the $N$ original vertices. Each time I turn on a vertex $w$ and union it with the already-on components touching it, a new component is born; I make a fresh internal node for that birth, tag it with the value $w$, and make it the parent of the component-roots that just merged (and of the leaf $w$ itself if $w$ was isolated). I'll create one internal node every time I turn on a vertex — even when $w$ has no on-neighbors yet and just sits alone — so that *every* vertex owns exactly one internal node carrying its threshold value $w$; that uniformity will matter when I query. After the whole sweep, I have a forest (a tree if the graph is connected) whose internal nodes are tagged with the vertex values $N-1, N-2, \dots$ in the order they were turned on.

Walk from a leaf up to the root. The internal nodes I pass through were created later and later in the sweep — and "later in the high-to-low sweep" means *smaller* vertex value. So the tags strictly decrease as I climb. Consider a query $(S, L)$. The set $V_L$ — reachable from $S$ using vertices $\ge L$ — is whatever component $S$ landed in by the time the threshold had only dropped to $L$, i.e. only vertices with value $\ge L$ had been turned on. In the tree, that's the subtree under the highest ancestor of $S$ whose tag is $\ge L$. Because tags decrease upward, I find it by climbing from $S$ while the parent's tag is still $\ge L$, and stopping the instant the parent's tag would dip below $L$. One subtlety: if $S$ itself is smaller than $L$, then $S$ isn't even in the "vertices $\ge L$" subgraph — it never got turned on by threshold $L$ — so $V_L$ is empty and the query is a definite no; the climb is only meaningful when $S \ge L$.

And the subtree under a node is a *contiguous range* if I number the leaves by a depth-first traversal of the tree — DFS order makes every subtree's leaves consecutive. So I run a DFS, assign each leaf a position $0,1,2,\dots$ in visitation order, and give every node the interval $[\text{lo},\text{hi}]$ of leaf-positions beneath it. Then $V_L$ is exactly the interval of the node I climbed to. The arbitrary scattered set has become an interval — the line-graph picture, recovered on a general graph by relabeling leaves along the tree.

The wolf side $V_R$ is the mirror image. There the constraint is $\le R$, so the subgraph grows as $R$ *increases*; I turn vertices on in *increasing* index order, $0,1,2,\dots$, unioning each new vertex with its already-on (smaller-numbered) neighbors, and build the analogous tree. Now tags increase as the sweep proceeds, so climbing from a leaf the tags still go the natural direction — I climb from $E$ while the parent's tag is $\le R$, and $V_R$ is the subtree I stop at, again a contiguous interval, but in *this* tree's DFS leaf order. (And if $E > R$, $E$ is outside the $\le R$ subgraph and $V_R$ is empty.)

The snag I have to respect is that the two trees order the leaves differently — the high tree's DFS visits vertices in one order, the low tree's DFS in another. So a vertex $v$ has two positions: $x_v$, its leaf-position in the high tree, and $y_v$, its leaf-position in the low tree. $V_L$ is an interval $[a_1,b_1]$ in the $x$-coordinate; $V_R$ is an interval $[a_2,b_2]$ in the $y$-coordinate. They live in different coordinate systems, so I can't just intersect two intervals on one line. But I *can* think of every original vertex as a point $(x_v, y_v)$ in the plane. The query "$V_L \cap V_R \ne \varnothing$" becomes: **is there any vertex-point whose $x$ lies in $[a_1,b_1]$ and whose $y$ lies in $[a_2,b_2]$?** A rectangle-nonempty test over $N$ fixed points. The two interval representations, fused into one 2D question.

Counting points in axis-aligned rectangles over a static point set is standard offline work. Let me lay the $N$ points out along the $x$-axis: define an array where position $x_v$ holds the value $y_v$ — so reading left to right along $x$, I see each point's $y$. A rectangle $[a_1,b_1]\times[a_2,b_2]$ counts points with $x\le b_1$ and $y\in[a_2,b_2]$, minus those with $x\le a_1-1$ and $y\in[a_2,b_2]$ — a difference of two "$x$-prefix, $y$-range" counts. So I sweep $x$ from $0$ upward, inserting each point's $y$ into a Fenwick tree (BIT) keyed by $y$. For each query I register two events: at $x = b_1$, a $+1$ event that asks "how many inserted points have $y\in[a_2,b_2]$ right now", and at $x = a_1-1$ (when $a_1>0$), a $-1$ event asking the same. Each event is a BIT range-sum, $O(\log N)$. After the sweep, a query's signed total is the number of vertices in its rectangle; positive means $V_L$ and $V_R$ share a vertex, i.e. a switch exists, i.e. yes. Queries that were already settled as empty ($S<L$ or $E>R$) are just no, skipped entirely.

Let me sanity-check the whole pipeline on a tiny example before I trust it, because the two sweep directions are easy to get backwards. Take the small graph with edges $5\!-\!1,\,1\!-\!2,\,1\!-\!3,\,3\!-\!4,\,3\!-\!0,\,5\!-\!2$ and the query $(S,E,L,R)=(4,2,1,2)$. By hand: from $4$, using only vertices $\ge 1$, I can reach $4,3,1,2,5$ (vertex $0$ is excluded, $\ge 1$) — that's $V_L$. From $2$, using only vertices $\le 2$, I can reach $2,1,0$ — that's $V_R$. Their intersection contains vertex $1$ and vertex $2$, both in $[1,2]$, so the answer is yes. And indeed vertex $2$: it's $\ge 1$ (reachable in human phase from $4$ via $4\!-\!3\!-\!1\!-\!2$) and $\le 2$ (it's the start of the wolf phase, trivially), so switching at $2$ works. Now the query $(4,2,2,2)$: human phase needs vertices $\ge 2$, so from $4$ I can only reach $\{4\}$ (its neighbor $3$ is $<2$), while $V_R$ from $2$ with $\le 2$ is $\{2,1,0\}$; disjoint, answer no. And $(5,4,3,4)$: human phase $\ge 3$ from $5$ gives $\{5\}$ (neighbors $1,2$ are $<3$); wolf $\le 4$ from $4$ gives $\{4,3,0,1,2\}$; disjoint, no. So this instance should print yes, no, no — that's what the two-tree intersection must reproduce.

I still need to be sure the high tree's climb is exact. When I turn vertices on from $N-1$ down to $L$, every merge that happens uses only vertices $\ge L$, and each merge is recorded as an internal node tagged with the value of the vertex that triggered it — a value that is $\ge L$ as long as we haven't dropped below threshold $L$. So the connected pieces present at threshold $L$ are precisely the subtrees hanging under the topmost nodes whose tags are still $\ge L$; the next node up was created at some threshold $< L$, i.e. by turning on a vertex $< L$, which is not allowed in $V_L$. Climbing while parent-tag $\ge L$ and stopping otherwise lands me on exactly that topmost in-range node, and its subtree leaves are exactly $V_L$. The low tree is the same argument with $\le R$.

One implementation detail matters for scale. If I climb parent pointers one edge at a time, a path-shaped merge history can make a query linear. The tags on a leaf-to-root path are monotone, so I can precompute $2^j$ ancestors. For the lower-bound side, a jump is valid exactly when the candidate ancestor still has tag at least $L$; for the upper-bound side, it is valid exactly when the candidate ancestor still has tag at most $R$. That makes each component-range lookup logarithmic. The code has two builds (decreasing order for the human/high side, increasing for the wolf/low side), each followed by DFS leaf positions, jump tables for the ancestor searches, and then one offline $x$-sweep over a Fenwick tree for the rectangle counts.

```python
import sys


class DSU:
    """Generic disjoint-set union over a fixed universe, with path compression.
    union(a, b) merges the sets; find(x) returns a set representative."""

    def __init__(self, size):
        self.par = list(range(size))

    def find(self, x):
        root = x
        while self.par[root] != root:
            root = self.par[root]
        while self.par[x] != root:
            self.par[x], x = root, self.par[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.par[rb] = ra
        return ra


def answer_queries(n, edges, queries):
    """edges: list of (u, v), 0-based undirected. queries: list of (S, E, L, R)
    with L <= R. For each query, return whether a walk from S to E exists that
    stays on vertices >= L up to a single switch vertex v (with L <= v <= R),
    then stays on vertices <= R. Returns a list of bools, one per query."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    def build_structure(order):
        # Turn vertices on in `order`; turning on w hangs the components of all
        # already-on neighbors (and the leaf w) under a fresh internal node
        # tagged with value w. Leaves 0..n-1 keep their own value as tag.
        cap = 2 * n
        parent = list(range(cap))          # forest parent (root points to self)
        dsu = DSU(cap)                     # current component representative
        children = [[] for _ in range(cap)]
        tag = list(range(cap))             # tag[node]: the vertex that created it
        nxt = n
        added = [False] * n
        for w in order:
            added[w] = True
            roots = [dsu.find(w)]
            seen = {roots[0]}
            for u in adj[w]:
                if added[u]:
                    r = dsu.find(u)
                    if r not in seen:
                        seen.add(r)
                        roots.append(r)
            node = nxt
            nxt += 1
            tag[node] = w
            for r in roots:
                parent[r] = node           # tree edge
                dsu.par[r] = node          # the old roots now live under node
            children[node] = roots
            dsu.par[node] = node           # node is the new component root

        # DFS leaf order: each subtree's leaves become a contiguous interval.
        left = [0] * nxt
        right = [-1] * nxt
        place = [0] * n
        timer = 0
        for root in sorted(x for x in range(nxt) if parent[x] == x):
            stack = [(root, 0)]
            while stack:                   # iterative post-order
                node, ci = stack[-1]
                if node < n:
                    place[node] = timer
                    left[node] = right[node] = timer
                    timer += 1
                    stack.pop()
                elif ci < len(children[node]):
                    stack[-1] = (node, ci + 1)
                    stack.append((children[node][ci], 0))
                else:
                    left[node] = min(left[c] for c in children[node])
                    right[node] = max(right[c] for c in children[node])
                    stack.pop()
        return parent[:nxt], tag[:nxt], left, right, place

    def make_jumps(parent):
        jumps = [parent[:]]
        bit = 1
        while (1 << bit) <= len(parent):
            prev = jumps[-1]
            jumps.append([prev[prev[x]] for x in range(len(parent))])
            bit += 1
        return jumps

    # human side: turn vertices on high-to-low; the threshold L is a lower bound.
    first_parent, first_tag, first_left, first_right, first_place = (
        build_structure(range(n - 1, -1, -1))
    )
    # wolf side: turn vertices on low-to-high; the threshold R is an upper bound.
    second_parent, second_tag, second_left, second_right, second_place = (
        build_structure(range(n))
    )
    first_jump = make_jumps(first_parent)
    second_jump = make_jumps(second_parent)

    def locate_first(S, L):
        if S < L:                          # S not even in the >=L subgraph
            return None
        x = S
        for jump in reversed(first_jump):
            y = jump[x]
            if y != x and first_tag[y] >= L:
                x = y                       # highest tag-in-range ancestor
        return first_left[x], first_right[x]

    def locate_second(E, R):
        if E > R:
            return None
        x = E
        for jump in reversed(second_jump):
            y = jump[x]
            if y != x and second_tag[y] <= R:
                x = y
        return second_left[x], second_right[x]

    # vertex v becomes the point (first_place[v], second_place[v]).
    axis_value = [0] * n
    for v in range(n):
        axis_value[first_place[v]] = second_place[v]

    Q = len(queries)
    usable = [False] * Q                   # False if a side is already empty
    pending = [[] for _ in range(n)]       # x-sweep events: (qid, sign, ylo, yhi)
    for qid, (S, E, L, R) in enumerate(queries):
        first = locate_first(S, L)
        second = locate_second(E, R)
        if first is None or second is None:
            continue
        xlo, xhi = first
        ylo, yhi = second
        usable[qid] = True
        if xlo > 0:
            pending[xlo - 1].append((qid, -1, ylo, yhi))
        pending[xhi].append((qid, +1, ylo, yhi))

    fenwick = [0] * (n + 1)                 # Fenwick over the y-axis

    def add_value(i):
        i += 1
        while i <= n:
            fenwick[i] += 1
            i += i & (-i)

    def prefix_value(i):                    # #inserted points with y in [0..i]
        i += 1
        total = 0
        while i > 0:
            total += fenwick[i]
            i -= i & (-i)
        return total

    hits = [0] * Q
    for x in range(n):                      # sweep x; insert this point's y
        add_value(axis_value[x])
        for (qid, sign, ylo, yhi) in pending[x]:
            count = prefix_value(yhi) - (prefix_value(ylo - 1) if ylo > 0 else 0)
            hits[qid] += sign * count       # rectangle count by x-prefix difference

    return [usable[qid] and hits[qid] > 0 for qid in range(Q)]


if __name__ == "__main__":
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it)); q = int(next(it))
    edges = [(int(next(it)), int(next(it))) for _ in range(m)]
    queries = [(int(next(it)), int(next(it)), int(next(it)), int(next(it)))
               for _ in range(q)]
    out = answer_queries(n, edges, queries)
    print("\n".join("YES" if a else "NO" for a in out))
```

I end up with this chain: a query is satisfiable iff the human-reachable set $V_L$ (from $S$ over vertices $\ge L$) and the wolf-reachable set $V_R$ (from $E$ over vertices $\le R$) share a vertex, and any shared vertex is automatically a legal switch. Per-query flooding is too slow, but the reachable sets are monotone — lowering $L$ or raising $R$ only merges components — so each merge history is a laminar family, hence a tree. If I build that tree by turning vertices on in decreasing order for the human side and increasing order for the wolf side, $V_L$ becomes the subtree under $S$'s highest ancestor with tag $\ge L$, and $V_R$ becomes the subtree under $E$'s highest ancestor with tag $\le R$; binary lifting finds those ancestors in logarithmic time, and in each tree's DFS leaf order those subtrees are contiguous intervals. The two trees order leaves differently, so I give each vertex a 2D point from its two positions, and the query "$V_L\cap V_R\ne\varnothing$" becomes "is the rectangle [human interval] $\times$ [wolf interval] nonempty" — which I answer for all queries together by one $x$-sweep with a Fenwick tree over the $y$-axis.
