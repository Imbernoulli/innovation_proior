# Heavy-light decomposition for path queries on a tree

## Problem

A tree on $n$ nodes, each holding a value. Support online: `update(u, val)` — set
the value at node $u$; and `path_query(u, v)` — the sum (or maximum) of the values
on the unique simple path between $u$ and $v$. With $n, q$ up to $\sim 10^5$ a
per-query walk of the path is $O(n)$ each, $O(nq)$ overall, too slow.

## Key idea

A segment tree answers point-update + range-aggregate in $O(\log n)$ — *if* the
path's nodes form a few contiguous array ranges. Plain DFS order makes subtrees
contiguous but scatters vertical ancestor-chains, which is exactly what a path is
made of. The fix is to choose the linear order.

**Heavy / light edges.** Root the tree. The **heavy child** of a node is the
child with the largest subtree (ties arbitrary); the edge to it is the **heavy
edge**, all other downward edges are **light**. Maximal runs of heavy edges are
**heavy chains**, and they partition the nodes (a node with no heavy child is a
length-one chain).

**The $O(\log n)$ chain bound.** Descending a light edge from $v$ into a
non-largest child $c$ at least halves the subtree size: the largest child has
subtree $\ge s(c)$, and both are disjoint inside $v$, so $s(v) \ge 2\,s(c)$, i.e.
$s(c) \le s(v)/2$. Subtree size starts at $n$ and bottoms out at $1$, so any
root-to-node path crosses at most $\log_2 n$ light edges — hence touches
$O(\log n)$ chains.

**Contiguous chains in one array.** A DFS that visits the heavy child *first*
gives every heavy chain a contiguous run of positions `pos[]` (chain head
smallest, bottom largest). Store `head[v]` = the top node of $v$'s chain. One
segment tree over the `pos` array then serves all updates and range queries; the
chains partition the $n$ nodes, so the array has exactly $n$ slots.

**Query by chain-climb (LCA for free).** To aggregate the path $u \to v$, while
$u$ and $v$ lie on different chains, lift the pointer whose chain *head is deeper*:
aggregate that chain's slice `[pos[head], pos[node]]`, then jump to
`parent[head]` (one light edge up). After $O(\log n)$ such steps the heads
coincide; the shallower node is the lowest common ancestor, and one final range
over `[pos[shallower], pos[deeper]]` finishes the path. The climb never assumes
which aggregate is used, so swapping the segment tree's combine to `max` answers
path-maximum unchanged.

## Algorithm

1. First DFS (iterative): `parent`, `depth`, subtree `size`, and `heavy[v]` =
   largest-subtree child of $v$.
2. Second DFS (iterative, heavy child first): `head[v]` and `pos[v]`, laying each
   chain into a contiguous block; build one segment tree over the values in `pos`
   order.
3. `update(u, val)`: point update at `pos[u]`.
4. `path_query(u, v)`: climb chains, always lifting the deeper-headed pointer,
   aggregating each chain slice; finish with the in-chain range through the LCA.

## Code

```python
import sys


def read_tree(data):
    """Parse n, the n node values, and n-1 edges (1-based in input) into a
    0-based adjacency list. Returns (n, values, adj)."""
    it = iter(data)
    n = int(next(it))
    values = [int(next(it)) for _ in range(n)]
    adj = [[] for _ in range(n)]
    for _ in range(n - 1):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        adj[u].append(v)
        adj[v].append(u)
    return n, values, adj


class SegmentTree:
    """Array-backed segment tree: point update, range query under an
    associative combine. Defaults to sum (swap _combine/_identity for max)."""

    def __init__(self, base):
        self.n = len(base)
        self.t = [0] * (2 * self.n)
        for i in range(self.n):
            self.t[self.n + i] = base[i]
        for i in range(self.n - 1, 0, -1):
            self.t[i] = self._combine(self.t[2 * i], self.t[2 * i + 1])

    @staticmethod
    def _combine(a, b):
        return a + b

    _identity = 0

    def update(self, i, val):
        i += self.n
        self.t[i] = val
        i >>= 1
        while i:
            self.t[i] = self._combine(self.t[2 * i], self.t[2 * i + 1])
            i >>= 1

    def query(self, l, r):
        """Aggregate over the index range [l, r] inclusive."""
        res = self._identity
        l += self.n
        r += self.n + 1
        while l < r:
            if l & 1:
                res = self._combine(res, self.t[l])
                l += 1
            if r & 1:
                r -= 1
                res = self._combine(res, self.t[r])
            l >>= 1
            r >>= 1
        return res


class TreePaths:
    def __init__(self, n, values, adj, root=0):
        self.n = n
        self.values = values
        self.adj = adj
        self.parent = [-1] * n
        self.depth = [0] * n
        self.size = [1] * n
        self.build(root)

    def build(self, root):
        n, adj = self.n, self.adj
        self.heavy = [-1] * n
        self.head = [0] * n
        self.pos = [0] * n
        parent, depth, size, heavy = self.parent, self.depth, self.size, self.heavy

        # First pass (iterative): parent, depth, subtree size, heavy child.
        order = []
        stack = [root]
        parent[root] = -1
        depth[root] = 0
        visited = [False] * n
        while stack:
            v = stack.pop()
            if visited[v]:
                continue
            visited[v] = True
            order.append(v)
            for c in adj[v]:
                if c != parent[v]:
                    parent[c] = v
                    depth[c] = depth[v] + 1
                    stack.append(c)
        for v in reversed(order):          # children before parents
            best = 0
            for c in adj[v]:
                if c != parent[v]:
                    size[v] += size[c]
                    if size[c] > best:
                        best = size[c]
                        heavy[v] = c

        # Second pass (iterative): heavy-child-first DFS assigns chain heads and
        # contiguous positions. Stack carries (vertex, chain head).
        head, pos = self.head, self.pos
        cur = 0
        stack = [(root, root)]
        while stack:
            v, h = stack.pop()
            head[v] = h
            pos[v] = cur
            cur += 1
            # Push light children first so the heavy child is processed next
            # (LIFO), keeping each heavy chain contiguous in pos.
            for c in adj[v]:
                if c != parent[v] and c != heavy[v]:
                    stack.append((c, c))
            if heavy[v] != -1:
                stack.append((heavy[v], h))

        base = [0] * n
        for v in range(n):
            base[pos[v]] = self.values[v]
        self.seg = SegmentTree(base)

    def update(self, u, val):
        self.seg.update(self.pos[u], val)

    def path_query(self, u, v):
        res = SegmentTree._identity
        head, pos, parent, depth = self.head, self.pos, self.parent, self.depth
        while head[u] != head[v]:
            if depth[head[u]] < depth[head[v]]:
                u, v = v, u
            res = SegmentTree._combine(res, self.seg.query(pos[head[u]], pos[u]))
            u = parent[head[u]]
        if depth[u] > depth[v]:
            u, v = v, u
        res = SegmentTree._combine(res, self.seg.query(pos[u], pos[v]))
        return res


def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    n, values, adj = read_tree(data)
    tp = TreePaths(n, values, adj)
    # (driver would read and dispatch queries here)


if __name__ == "__main__":
    main()
```

## Complexity

- **Build:** $O(n)$ — two linear DFS passes (iterative, so $n = 10^5$ bamboos do
  not overflow the stack) plus an $O(n)$ segment-tree build. $O(n)$ memory.
- **`update`:** $O(\log n)$ — one segment-tree point update.
- **`path_query`:** $O(\log^2 n)$ — $O(\log n)$ chains crossed, each an
  $O(\log n)$ range query.
- Total over $q$ operations: $O\big(n + q\log^2 n\big)$, ample for
  $n, q \sim 10^5$. Switching the segment tree's combine from sum to max turns the
  same structure into path-maximum with no other change.
