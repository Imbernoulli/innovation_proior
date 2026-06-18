# Context

## Problem

Given a rooted tree of $n$ nodes, answer $q$ queries each asking the Lowest Common Ancestor of two nodes $u$ and $v$, with $O(n \log n)$ preprocessing and $O(1)$ time per query. ($n, q$ up to $\sim 10^5 / 10^6$.)

The tree is rooted at node $0$. Input is 1-based: `n q`, then `n - 1` undirected tree edges, then `q` query pairs. The reusable query method works with 0-based node ids; the command-line driver prints 1-based answers.

## Code framework

The tree is parsed into a 0-based adjacency list. A generic immutable-array minimum helper is available; it stores positions of minima so callers can recover whatever value they keep at those positions. Fill the one-time tree preparation and the query method.

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
        # TODO
        pass

    def lca(self, u, v):
        # TODO
        pass


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
