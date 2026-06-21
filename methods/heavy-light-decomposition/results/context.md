# Context

## Problem

Given a tree of $n$ nodes each holding a value, support two online operations:
update the value at a node, and query the sum (or maximum) of values on the path
between two given nodes $u$ and $v$ — fast enough for $n, q$ up to $\sim 10^5$.

The operations arrive online: each must be answered before the next is read, so
the queries cannot be reordered or batched offline. The path between $u$ and $v$
is the unique simple path in the tree, and it can contain anywhere from one to
$n$ nodes.

## Code framework

The tree is read into a $0$-based adjacency list together with the per-node
values. A generic segment tree over a flat array is already available — point
update plus range query under any associative combine (it is written for sum
below; swapping `_combine`/`_identity` turns it into a max tree). To be filled
in are a `build` that preprocesses the tree and the two top-level operations
`update(u, val)` and `path_query(u, v)`.

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
        # TODO
        pass

    def update(self, u, val):
        # TODO
        pass

    def path_query(self, u, v):
        # TODO
        pass


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
