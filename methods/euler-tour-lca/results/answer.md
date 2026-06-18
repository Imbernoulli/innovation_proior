# LCA by Euler Tour and Sparse-Table RMQ

## Problem

For a rooted tree, preprocess once so each lowest-common-ancestor query is answered in `O(1)` time. The tree is static, so the preprocessing may spend `O(n log n)` time and space.

## Method

Run one depth-first traversal from the root and record a node every time the walk touches it: once on entry and again whenever the traversal returns to it from a child. This gives an Euler array of length `2n - 1`. Store a parallel depth array, and store `first[u]`, the first index where node `u` appears in the Euler array.

For two nodes `u` and `v`, let `left, right = sorted((first[u], first[v]))`. In the Euler slice from `left` to `right`, the traversal moves from the first node's side of the tree up through the lowest common ancestor and then down toward the second node's side. It never has to go above that ancestor, while every detour inside a child subtree is deeper. Therefore the answer is the node at the minimum-depth position in that slice:

$$
\mathrm{LCA}(u, v) =
\mathrm{euler}\left[
  \arg\min_{left \le i \le right} \mathrm{depth\_at}[i]
\right].
$$

The remaining task is static range minimum over the depth array. A sparse table stores, for each power-of-two block, the position whose depth is minimal. For a query range, choose `k = floor(log2(length))` and compare the two overlapping blocks `[left, left + 2^k - 1]` and `[right - 2^k + 1, right]`. The overlap is harmless because minimum is idempotent. This gives two table reads and one final Euler-array read.

## Code

```python
import sys


def read_input(data):
    """Parse n, q, n-1 edges, and q queries from 1-based input."""
    it = iter(data)
    n = int(next(it))
    q = int(next(it))
    adj = [[] for _ in range(n)]
    for _ in range(n - 1):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        adj[u].append(v)
        adj[v].append(u)
    queries = []
    for _ in range(q):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        queries.append((u, v))
    return n, adj, queries


class SparseTableMin:
    """Range minimum over a fixed integer array.

    Each table cell stores the position whose key is minimal, so a caller can
    recover the value associated with that position. Build is O(m log m);
    queries are O(1).
    """

    def __init__(self, key):
        self.key = key
        m = len(key)
        self.log = [0] * (m + 1)
        for i in range(2, m + 1):
            self.log[i] = self.log[i >> 1] + 1

        levels = self.log[m] + 1 if m else 1
        table = [[0] * m for _ in range(levels)]
        table[0] = list(range(m))
        for k in range(1, levels):
            half = 1 << (k - 1)
            span = half << 1
            row = table[k]
            prev = table[k - 1]
            for start in range(m - span + 1):
                left = prev[start]
                right = prev[start + half]
                row[start] = left if key[left] <= key[right] else right
        self.table = table

    def argmin(self, left, right):
        """Return a position of the minimum key in inclusive range [left, right]."""
        length = right - left + 1
        k = self.log[length]
        a = self.table[k][left]
        b = self.table[k][right - (1 << k) + 1]
        return a if self.key[a] <= self.key[b] else b


class LCA:
    def __init__(self, n, adj, root=0):
        self.n = n
        self.adj = adj
        self.preprocess(root)

    def preprocess(self, root):
        euler = []
        depth_at = []
        first = [-1] * self.n
        seen = [False] * self.n

        stack = [(root, 0, 0)]
        seen[root] = True
        first[root] = 0
        euler.append(root)
        depth_at.append(0)

        while stack:
            node, depth, child_index = stack[-1]
            children = self.adj[node]
            advanced = False

            while child_index < len(children):
                nxt = children[child_index]
                child_index += 1
                if seen[nxt]:
                    continue

                stack[-1] = (node, depth, child_index)
                seen[nxt] = True
                first[nxt] = len(euler)
                euler.append(nxt)
                depth_at.append(depth + 1)
                stack.append((nxt, depth + 1, 0))
                advanced = True
                break

            if not advanced:
                stack.pop()
                if stack:
                    parent, parent_depth, _ = stack[-1]
                    euler.append(parent)
                    depth_at.append(parent_depth)

        self.euler = euler
        self.first = first
        self.depth_at = depth_at
        self.rmq = SparseTableMin(depth_at)

    def lca(self, u, v):
        left = self.first[u]
        right = self.first[v]
        if left > right:
            left, right = right, left
        pos = self.rmq.argmin(left, right)
        return self.euler[pos]


def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    n, adj, queries = read_input(data)
    solver = LCA(n, adj)
    out = [str(solver.lca(u, v) + 1) for u, v in queries]
    sys.stdout.write("\n".join(out))


if __name__ == "__main__":
    main()
```

## Complexity

The traversal builds arrays of length `2n - 1`. The sparse table has `O(log n)` levels, so preprocessing costs `O(n log n)` time and space. Each query reads `first[u]`, `first[v]`, does one sparse-table range-minimum query over the depth array, and returns one Euler-array entry, so query time is `O(1)`.
