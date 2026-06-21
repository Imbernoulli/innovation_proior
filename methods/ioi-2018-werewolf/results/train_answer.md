We are handed an undirected graph on $N$ vertices, numbered $0$ through $N-1$, with $M$ edges, and a pile of $Q$ queries. Each query $(S,E,L,R)$ with $L\le R$ asks a yes/no question about a walk from $S$ to $E$ that runs in two phases: at the start we may only step on vertices whose index is at least $L$, then at exactly one moment we flip a switch at some vertex, and from that vertex onward we may only step on vertices whose index is at most $R$. The switch vertex itself must be legal for both halves, so it lives in $[L,R]$. The literal way to decide one query is a two-color BFS over a $2N$-state graph — states $(v,0)$ "at $v$, still in phase one" and $(v,1)$ "at $v$, already switched", with intra-phase edges gated by $\ge L$ and $\le R$ and a free switch edge $(v,0)\to(v,1)$ whenever $L\le v\le R$ — and then asking whether $(E,1)$ is reachable from $(S,0)$. That is correct and runs in $O(N+M)$ per query, but with both $N$ and $Q$ up to a few hundred thousand, $O(Q(N+M))$ is hopeless. The queries must share preprocessing instead of each touching the whole graph.

The first move is to strip the two-color trick down to its essence. Phase one is a walk from $S$ that never leaves the set of vertices with index $\ge L$, so the switch vertex must lie in the connected component of $S$ in the subgraph induced by $\{v:v\ge L\}$ — call it $V_L$. Phase two is a walk from the switch vertex to $E$ over vertices $\le R$; since the graph is undirected, that is the same as the component of $E$ in the subgraph induced by $\{v:v\le R\}$ — call it $V_R$. A vertex in $V_L$ already has index $\ge L$ and a vertex in $V_R$ already has index $\le R$, so any vertex in $V_L\cap V_R$ automatically lies in $[L,R]$ and is a legal switch; the explicit switch constraint comes for free. The entire question therefore collapses to whether $V_L$ and $V_R$ share a vertex, with two cheap edge cases: if $S<L$ then $V_L=\varnothing$ and the answer is no, and if $E>R$ then $V_R=\varnothing$ and the answer is no. Recomputing both sets per query is still two BFS floods, $O(N+M)$ each, and the waste is that nearby thresholds produce almost identical sets. We want to compute these reachable sets once, as a structure, and then read each query off it.

I propose to solve this with two Kruskal reconstruction trees feeding a single offline 2D range-intersection sweep. The lever is monotonicity: as $L$ decreases, the subgraph "$\ge L$" only gains vertices and edges, so components only ever merge, never split. That is an incremental-union process, exactly what disjoint-set union is built for. Sweep $L$ from high to low, turning vertices on in decreasing index order; when vertex $w$ turns on, union it with every already-on neighbor (those are precisely the neighbors with larger index, which turned on earlier). The merge history of such a sweep is a laminar family — any two components ever seen are either disjoint or nested, since lowering the threshold only fuses — and a laminar family on $N$ leaves is a tree. So rather than throw the merge history away, build it explicitly. The leaves are the $N$ original vertices, each tagged with its own value. Each time we turn on a vertex $w$, we create one fresh internal node tagged with $w$ and make it the parent of the current component-roots of all already-on neighbors together with the leaf $w$; crucially we mint a new internal node for *every* vertex turned on, even an isolated one with no on-neighbors yet, so that every vertex owns exactly one internal node carrying its threshold value — a uniformity that makes the query climb clean.

Climbing from a leaf to the root, the internal nodes were created later and later in the high-to-low sweep, and "later" means *smaller* tag, so tags strictly decrease upward. For a query $(S,L)$, the set $V_L$ is whatever component $S$ landed in once the threshold had only dropped to $L$, i.e. only vertices with value $\ge L$ had turned on; in the tree that is the subtree under the highest ancestor of $S$ whose tag is still $\ge L$. Because tags decrease upward, we find it by climbing while the parent's tag is $\ge L$ and stopping the instant it would dip below $L$. The decisive payoff is that a subtree's leaves form a *contiguous interval* if we number leaves by a DFS traversal of the tree. So after building the tree we run a DFS, assign each leaf a position $0,1,2,\dots$ in visitation order, and give each node the interval $[\mathrm{lo},\mathrm{hi}]$ of leaf-positions beneath it. The arbitrary scattered set $V_L$ becomes an interval $[a_1,b_1]$ — the easy line-graph picture, recovered on a general graph by relabeling leaves along the tree.

The other side is the mirror image. The constraint there is $\le R$, so the subgraph grows as $R$ increases; we turn vertices on in *increasing* index order, union each with its already-on smaller-numbered neighbors, and build the analogous tree, whose tags increase along the sweep. We climb from $E$ while the parent's tag is $\le R$, and $V_R$ is the subtree we stop at — again a contiguous interval $[a_2,b_2]$, but in *this* tree's DFS leaf order. The snag is that the two trees order the leaves differently, so each vertex $v$ has two positions: $x_v$ in the high tree and $y_v$ in the low tree. $V_L$ is the interval $[a_1,b_1]$ in the $x$-coordinate and $V_R$ is the interval $[a_2,b_2]$ in the $y$-coordinate, living in different coordinate systems, so they cannot be intersected on one line. The resolution is to view each original vertex as a point $(x_v,y_v)$ in the plane. Then $V_L\cap V_R\ne\varnothing$ is exactly the question of whether any vertex-point falls in the axis-aligned rectangle
$$[a_1,b_1]\times[a_2,b_2],$$
a rectangle-nonempty test over $N$ fixed points.

Counting points in axis-aligned rectangles over a static point set is standard offline work. Lay the $N$ points along the $x$-axis — an array whose position $x_v$ holds the value $y_v$ — and sweep $x$ upward, inserting each point's $y$ into a Fenwick tree (BIT) keyed by $y$. A rectangle count is a difference of two "$x$-prefix, $y$-range" counts: the number of points with $x\le b_1$ and $y\in[a_2,b_2]$ minus the number with $x\le a_1-1$ and $y\in[a_2,b_2]$. So for each query we register two events, a $+1$ at $x=b_1$ and (when $a_1>0$) a $-1$ at $x=a_1-1$, each asking the BIT for the count of inserted $y$ in $[a_2,b_2]$. After the sweep, a query's signed total is the number of its vertices in the rectangle; positive means $V_L$ and $V_R$ share a vertex, hence yes. Queries already settled empty by $S<L$ or $E>R$ are simply no. For scale, the upward climb in each tree must not walk parent pointers one edge at a time, since a path-shaped merge history would make a query linear; because the tags along any leaf-to-root path are monotone, we precompute $2^j$-ancestor jump tables (binary lifting) and a jump is valid exactly when the candidate ancestor still has tag $\ge L$ on the lower-bound side or $\le R$ on the upper-bound side, making each component-range lookup logarithmic. The whole pipeline runs in $O\big((N+M)\,\alpha(N)+(N+Q)\log N\big)$ time.

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
