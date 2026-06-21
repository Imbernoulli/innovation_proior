# Context

## Problem

Given a static array of `n` integers and `q` queries `(l, r, k)`, report the
`k`-th smallest value in `a[l..r]`. The array is fixed after input; queries do
not update it. Indices are 1-based, `1 <= l <= r <= n`, and `k` is 1-based with
`1 <= k <= r - l + 1`. Equal values count with multiplicity.

## Code framework

The scaffold below already handles input, coordinate compression, node storage,
and output. Values are mapped to a dense domain `1..self.len` using the sorted
distinct values in `self.vals`; a returned value index is mapped back to the
original value at the end. The counting structure uses dynamic nodes in parallel
arrays: `self.sum[node]` is the count stored at a node, while `self.ls[node]`
and `self.rs[node]` are child node ids, with node `0` the shared empty node.

A plain value-indexed count segment tree over `1..self.len` is provided:
`insert(value_index)` adds one occurrence by walking the root-to-leaf path and
bumping each `sum`, and `count(node, l, r, lo, hi)` reports how many inserted
items fall in a value sub-range. On top of it, an order-statistic descent finds
the `k`-th smallest of the items currently in the tree: at each node compare `k`
against the left child's `sum`, walk left, or walk right after subtracting it.
The scaffold exposes `build()` for preprocessing the array and `kth(l, r, k)` as the top-level query entry point.

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
        self.troot = 0          # root of the plain count tree
        self.build()

    def getid(self, x):
        return bisect_left(self.vals, x) + 1

    def _new(self, s, l, r):
        self.sum.append(s)
        self.ls.append(l)
        self.rs.append(r)
        return len(self.sum) - 1

    def insert(self, k, l, r, node):
        """Add one occurrence of value index k to the count tree; bump sums
        along the root-to-leaf path. Returns the (possibly new) node id."""
        if node == 0:
            node = self._new(0, 0, 0)
        self.sum[node] += 1
        if l == r:
            return node
        mid = (l + r) >> 1
        if k <= mid:
            self.ls[node] = self.insert(k, l, mid, self.ls[node])
        else:
            self.rs[node] = self.insert(k, mid + 1, r, self.rs[node])
        return node

    def count(self, node, l, r, lo, hi):
        """Number of inserted items with value index in [lo, hi]."""
        if node == 0 or hi < l or r < lo:
            return 0
        if lo <= l and r <= hi:
            return self.sum[node]
        mid = (l + r) >> 1
        return (self.count(self.ls[node], l, mid, lo, hi)
                + self.count(self.rs[node], mid + 1, r, lo, hi))

    def build(self):
        # TODO
        pass

    def kth(self, l, r, k):
        """k-th smallest value in a[l..r] (1-based l, r; 1-based k)."""
        # TODO
        pass


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
