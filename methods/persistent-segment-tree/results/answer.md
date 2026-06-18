# Persistent Segment Tree for Static Range K-th Smallest

## Method

Compress the array values to a dense value domain `1..len`, then store a
value-indexed count segment tree for each array prefix. The tree for prefix `i`
contains the multiset `a[1..i]`. A query range `a[l..r]` is represented by the
difference between the prefix tree at `r` and the prefix tree at `l - 1`.

The query descends through those two roots at the same time. At each internal
value interval, compute

```text
x = sum[left child of root[r]] - sum[left child of root[l - 1]]
```

This is the number of range elements in the lower value half. If `k <= x`, the
answer is in that lower half; otherwise the answer is in the upper half with
rank `k - x`. The reached leaf is the compressed value index.

The prefix trees are stored persistently by path copying. Inserting one value
changes only one root-to-leaf path, so each new prefix root allocates
`O(log n)` nodes and shares every untouched subtree with the previous prefix.
Thus preprocessing uses `O(n log n)` time and memory, and each query costs
`O(log n)`.

## Code

```python
import sys
from bisect import bisect_left


class RangeKth:
    def __init__(self, a):
        self.a = a
        self.n = len(a)
        self.vals = sorted(set(a))
        self.len = len(self.vals)
        self.sum = [0]
        self.ls = [0]
        self.rs = [0]
        self.root = [0] * (self.n + 1)
        self.build()

    def getid(self, x):
        return bisect_left(self.vals, x) + 1

    def _new(self, s, l, r):
        self.sum.append(s)
        self.ls.append(l)
        self.rs.append(r)
        return len(self.sum) - 1

    def _insert(self, k, l, r, prev):
        # Copy this node; reuse both of prev's subtrees by default.
        cur = self._new(self.sum[prev] + 1, self.ls[prev], self.rs[prev])
        if l == r:
            return cur
        mid = (l + r) >> 1
        # Rebuild only the half holding k; the other child stays shared.
        if k <= mid:
            self.ls[cur] = self._insert(k, l, mid, self.ls[cur])
        else:
            self.rs[cur] = self._insert(k, mid + 1, r, self.rs[cur])
        return cur

    def build(self):
        # root[0] = empty (node 0); each prefix copies one root-to-leaf path.
        for i in range(1, self.n + 1):
            self.root[i] = self._insert(
                self.getid(self.a[i - 1]), 1, self.len, self.root[i - 1]
            )

    def _query(self, u, v, l, r, k):
        # u = root[l-1], v = root[r]; counts over a[l..r] are v minus u.
        if l == r:
            return l
        mid = (l + r) >> 1
        x = self.sum[self.ls[v]] - self.sum[self.ls[u]]   # window items in lower half
        if k <= x:
            return self._query(self.ls[u], self.ls[v], l, mid, k)
        return self._query(self.rs[u], self.rs[v], mid + 1, r, k - x)

    def kth(self, l, r, k):
        """k-th smallest value in a[l..r] (1-based l, r; 1-based k)."""
        idx = self._query(self.root[l - 1], self.root[r], 1, self.len, k)
        return self.vals[idx - 1]


def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    q = int(next(it))
    a = [int(next(it)) for _ in range(n)]
    rk = RangeKth(a)
    out = []
    for _ in range(q):
        l = int(next(it)); r = int(next(it)); k = int(next(it))
        out.append(str(rk.kth(l, r, k)))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()
```

## Complexity

Preprocessing performs one path-copying insert per array element, so it takes
`O(n log n)` time and stores `O(n log n)` nodes. Each query performs one paired
descent through the value domain, so it takes `O(log n)` time.
