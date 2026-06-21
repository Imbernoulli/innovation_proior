I am given a tree on $n$ nodes, each carrying a value, and two operations that arrive online and must each be answered before the next is read: `update(u, val)` sets the value at node $u$, and `path_query(u, v)` returns the aggregate — the sum, or just as well the maximum — of the values on the unique simple path between $u$ and $v$, with $n$ and the number of operations both up to about $10^5$. The honest starting point is to walk the path: find the route from $u$ to $v$ and add up the node values. But a single path can sweep through $\Theta(n)$ nodes — on a bamboo, a chain of $n$ nodes, the path from one end to the other is the whole tree — so one query costs $O(n)$ and $q$ of them cost $O(nq)\approx 10^{10}$, which is hopeless. And because updates interleave with queries, I cannot precompute all the answers offline either. The thing that kills me is that I re-sum $\Theta(n)$ nodes from scratch on every query.

For a query over a contiguous *range* of an array I know exactly what to reach for — a segment tree, which does point update and range aggregate each in $O(\log n)$. So the real question is not "how do I sum a path" but "can I lay the tree's nodes into a linear array so that any path becomes a few contiguous slices?" The natural candidate, DFS-visit order `pos[v]`, has one beautiful property: every *subtree* occupies a contiguous block, because DFS enters a subtree, finishes it entirely, and only then leaves. But a path is not a subtree. Split it at the lowest common ancestor $l$ of $u$ and $v$ and it becomes two upward ancestor-chains, $u\to l$ and $v\to l$ — and in plain DFS order the ancestors of a node are scattered all over `pos`, so a vertical chain is in general $\Theta(\text{depth})$ separate positions, not a contiguous run. Plain DFS order gives subtrees for free and does nothing for the vertical chains a path is actually made of. I need a *different* linear order, one tuned so that the chains I query are contiguous.

The method is heavy-light decomposition. Suppose I cut the tree into top-to-bottom chains and lay each chain into a contiguous block of one flat array; then an upward path $u\to l$ crosses some number of chains, and on each crossed chain I do one segment-tree range query, so the per-query cost is (number of chains crossed) $\times\;O(\log n)$. Everything hinges on bounding the number of chains a single root-to-node path can cross — and that number is entirely at my mercy, because *I* choose which downward edges stay within a chain and which start a new one. At each node I let exactly one child's edge continue the current chain and declare the rest chain boundaries; the entire game is to choose the continuing child so that *no* root-to-node descent crosses more than $O(\log n)$ boundaries, no matter the tree's shape. Choosing blindly fails — an adversary nests the wrong choices and a descent zig-zags across $\Theta(n)$ chains — so the choice must be tied to subtree size. The rule is: the continuing child is the one with the **largest subtree**. Call the edge to it the *heavy* edge and the rest *light* edges; maximal runs of heavy edges are *heavy chains*, and since each node has exactly one parent and at most one heavy edge going down, the heavy edges decompose into disjoint top-to-bottom chains that partition all $n$ nodes (a node with no heavy child is a chain of length one).

What this buys is a clean halving bound. When a descent takes a light edge from $v$ into a child $c$ that is *not* the largest, the largest child has subtree at least $s(c)$, and the two children are disjoint pieces of $v$'s subtree, so $s(v)\ge s(c)+s(\text{largest})\ge 2\,s(c)$, giving

$$s(c)\le \frac{s(v)}{2}.$$

Every light step at least halves the subtree size of where I am standing; the size starts at $n$ at the root and cannot drop below $1$, so any root-to-node path takes at most $\log_2 n$ light steps and therefore lies on at most $O(\log n)$ chains. The bound does not care about the tree's shape — that is exactly why the adversary cannot beat the largest-subtree rule, where any blind rule fails. The tie case is harmless: if two children are equal-largest, only one becomes heavy, and the other is then a non-largest child with subtree $\le s(v)/2$, so the halving still holds.

To make each chain a contiguous slice of the array I run a DFS that visits the **heavy child first**, before any light children. Then walking down a node, its heavy child, that child's heavy child, and so on, I assign consecutive `pos` values straight down a whole heavy chain before ever backing up to a light branch — so each chain becomes an interval `[pos[head], pos[bottom]]`, head smallest and bottom largest. I store `head[v]`, the topmost node of $v$'s chain, so from any node I can see which chain it is on and jump to its top. Because the chains partition the $n$ nodes, the array has exactly $n$ slots and a single $O(n)$ segment tree over it serves every update and every range query — one tree, not one per chain. `update(u, val)` is then a single point update at `pos[u]`, $O(\log n)$.

The path query climbs chain by chain, and the lowest common ancestor falls out for free. I keep two pointers $u$ and $v$. While they sit on different chains (`head[u] != head[v]`), I lift the one whose chain *head is deeper*: that node's chain head, being deeper, cannot be an ancestor of the other pointer, so the slice of the path from the chain head down to the pointer — `query(pos[head[u]], pos[u])` — is definitely part of the answer and can be aggregated now, after which I jump that pointer to `parent[head[u]]`, stepping up one light edge onto the next chain. Each iteration moves a pointer up one entire chain, so the loop runs $O(\log n)$ times by the light-edge bound. When the heads finally coincide, both pointers lie on one chain; the shallower of them is the lowest common ancestor $l$, and a final `query` over `[pos[shallower], pos[deeper]]` adds the last stretch through $l$ — I never computed the LCA on the side. Accounting confirms no double-count and no gap: while heads differ I consume `[pos[head[u]], pos[u]]` and then jump strictly above it to the parent of the head, so the consumed and remaining parts abut without overlap; when the heads match the inclusive range covers $l$ exactly once, and the two arms stitched at $l$ count it once. The climb never assumes anything about the aggregate, so swapping the segment tree's `_combine`/`_identity` from sum to max returns the path maximum with no other change. The cost is $O(\log n)$ chains times $O(\log n)$ per range query, i.e. $O(\log^2 n)$ per query, with $O(n)$ build and $O(\log n)$ updates — over $q$ operations, $O(n + q\log^2 n)$, ample at $n,q\sim 10^5$. The extra $\log$ over a single-$\log$ trick is the price of vertical chains under a general associative aggregate; prefix-aggregate-per-chain shortcuts exist but need an invertible or static aggregate, whereas I want plain online point-update with sum *or* max.

One implementation hazard at $n=10^5$: a recursive DFS would recurse to depth $n$ on a bamboo and overflow the interpreter stack, so both passes use explicit stacks. The first pass pushes the root, records nodes while setting `parent`/`depth` on the way down, then walks that visit order in reverse — children before parents — to accumulate subtree sizes and pick each node's heavy child as the running-largest child. The decompose pass carries `(vertex, chain_head)` on its stack and, to make the heavy child come off next and keep its chain contiguous, pushes the light children first and the heavy child last, exploiting the stack's LIFO order. Then the array is filled in `pos` order and handed to the segment tree.

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
