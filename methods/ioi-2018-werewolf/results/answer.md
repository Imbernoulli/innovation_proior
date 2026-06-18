# Kruskal reconstruction trees + 2D range-intersection for the two-phase reachability query

## Problem

An undirected graph on $N$ vertices ($0..N-1$), $M$ edges, and $Q$ queries
$(S,E,L,R)$ with $L \le R$. A query asks whether some walk goes $S \to E$ in two
phases: it stays on vertices with index $\ge L$ until a single switch vertex $v$
(which must satisfy $L \le v \le R$), then stays on vertices with index $\le R$
to the end. Answer each query yes/no.

## Key idea

**Reduce a query to a set intersection.** Let
$V_L$ = vertices reachable from $S$ using only vertices $\ge L$ (the component of
$S$ in the subgraph induced by $\{v \ge L\}$), and
$V_R$ = vertices reachable from $E$ using only vertices $\le R$ (the component of
$E$ in the subgraph induced by $\{v \le R\}$; the graph is undirected, so
reaching $E$ from a switch vertex equals reaching the switch vertex from $E$). A
valid switch vertex is exactly a vertex in $V_L \cap V_R$ — and any such vertex
has index $\ge L$ and $\le R$ automatically, so the answer is **yes iff
$V_L \cap V_R \neq \varnothing$**. (If $S < L$ then $V_L = \varnothing$; if
$E > R$ then $V_R = \varnothing$ — those queries are immediately no.)

**Make the reachable sets contiguous via two Kruskal reconstruction trees.** As
$L$ decreases the subgraph "$\ge L$" only gains vertices, so components only
merge — a laminar (nested-or-disjoint) family, i.e. a tree. Build it: turn
vertices on in **decreasing** index order; turning on $w$ creates a fresh
internal node tagged $w$, made the parent of the components of all already-on
neighbors (and of the leaf $w$). Tags strictly decrease from any leaf to the
root, so $V_L$ is the subtree under the **highest ancestor of $S$ whose tag is
$\ge L$**, found in logarithmic time with binary lifting over parent pointers. In DFS leaf order,
that subtree is a **contiguous interval** $[a_1, b_1]$. The wolf side is the
mirror: turn vertices on in **increasing** index order; $V_R$ is the subtree
under the highest ancestor of $E$ with tag $\le R$, a contiguous interval
$[a_2, b_2]$ in *that* tree's leaf order.

**Intersect via a 2D rectangle test.** The two trees order leaves differently,
so vertex $v$ gets two coordinates: $x_v = \text{posH}[v]$ (high-tree leaf
index) and $y_v = \text{posL}[v]$ (low-tree leaf index). The query
"$V_L \cap V_R \neq \varnothing$" becomes: **is there a vertex-point with
$x \in [a_1,b_1]$ and $y \in [a_2,b_2]$?** — a rectangle-nonempty test over $N$
static points. Answer all queries offline by sweeping $x$ and maintaining a
Fenwick tree (BIT) over $y$: a rectangle count is the $x$-prefix difference
$\text{cnt}(b_1) - \text{cnt}(a_1-1)$, each a $y$-range sum in the BIT.

## Algorithm

1. Build the **high tree** (vertices on in decreasing order) and **low tree**
   (increasing order); each via a DSU that records the merge tree.
2. DFS each tree to give every leaf a position and every node a contiguous
   leaf-interval $[\text{lo},\text{hi}]$.
3. Build binary-lifting tables for both trees. For each query, jump upward from
   $S$ in the high tree while the candidate ancestor's tag is $\ge L$ to get
   $[a_1,b_1]$, and from $E$ in the low tree while the candidate ancestor's tag
   is $\le R$ to get $[a_2,b_2]$; if $S<L$ or $E>R$, answer no.
4. Lay vertices out as points $(\text{posH},\text{posL})$ and answer every
   rectangle query with one $x$-sweep over a BIT on the $y$-axis.

## Code

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

## Complexity

- Building the two trees and their DFS leaf orders touches each vertex and edge a
  constant number of times through a near-linear DSU; binary-lifting tables add
  $O(N\log N)$ preprocessing.
- Each query does two logarithmic ancestor searches and contributes two Fenwick
  events. The sweep performs $O(N+Q)$ Fenwick operations, each $O(\log N)$.
- Total $O\big((N+M)\,\alpha(N) + (N+Q)\log N\big)$ time and
  $O(N\log N + M + Q)$ memory.
