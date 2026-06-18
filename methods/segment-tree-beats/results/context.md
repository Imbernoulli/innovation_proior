# Context

## Problem

Maintain an integer array `a[1..n]` under online range operations:

1. `chmin l r x`: for every `i` in `[l, r]`, set `a[i] = min(a[i], x)`;
2. `max l r`: report `max(a[l..r])`;
3. `sum l r`: report `sum(a[l..r])`.

The array and the operation stream can both be very large, so scanning a range for
each operation is too slow. A useful implementation must keep the usual segment
tree shape, answer the two read-only queries by merging node aggregates, and make
range `chmin` avoid descending to every covered leaf whenever the node summary is
strong enough to handle the update locally.

## Code framework

The input loop, recursive segment tree, range splitting, and query traversal are
ordinary infrastructure. The open part is the node summary and the range-update
logic that decides how much of a covered node can be handled before descending.

```python
import sys
from sys import setrecursionlimit

NEG = float("-inf")


class RangeChminTree:
    def __init__(self, a):
        self.n = len(a)
        size = 4 * self.n
        self.slot_a = [0] * size
        self.slot_b = [NEG] * size
        self.slot_c = [0] * size
        self.sm = [0] * size
        self.tag = [NEG] * size
        self._build(1, 0, self.n - 1, a)

    def _pushup(self, u):
        # TODO: recombine this node's summary from its children.
        pass

    def _apply(self, u, x):
        # TODO: fold a full-node range update into this node.
        pass

    def _pushdown(self, u):
        if self.tag[u] == NEG:
            return
        self._apply(u << 1, self.tag[u])
        self._apply(u << 1 | 1, self.tag[u])
        self.tag[u] = NEG

    def _build(self, u, l, r, a):
        if l == r:
            self.sm[u] = a[l]
            # TODO: initialize this leaf's summary.
            return
        mid = (l + r) >> 1
        self._build(u << 1, l, mid, a)
        self._build(u << 1 | 1, mid + 1, r, a)
        self._pushup(u)

    def chmin(self, ql, qr, x, u=1, l=0, r=None):
        if r is None:
            r = self.n - 1
        # TODO: complete the range update.
        pass

    def query_max(self, ql, qr, u=1, l=0, r=None):
        if r is None:
            r = self.n - 1
        if ql <= l and r <= qr:
            return self._node_max(u)
        mid = (l + r) >> 1
        self._pushdown(u)
        res = NEG
        if ql <= mid:
            res = max(res, self.query_max(ql, qr, u << 1, l, mid))
        if mid < qr:
            res = max(res, self.query_max(ql, qr, u << 1 | 1, mid + 1, r))
        return res

    def query_sum(self, ql, qr, u=1, l=0, r=None):
        if r is None:
            r = self.n - 1
        if ql <= l and r <= qr:
            return self.sm[u]
        mid = (l + r) >> 1
        self._pushdown(u)
        res = 0
        if ql <= mid:
            res += self.query_sum(ql, qr, u << 1, l, mid)
        if mid < qr:
            res += self.query_sum(ql, qr, u << 1 | 1, mid + 1, r)
        return res

    def _node_max(self, u):
        # TODO: return this node's read-only aggregate.
        pass


def main():
    setrecursionlimit(1 << 20)
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    out = []
    T = int(next(it))
    for _ in range(T):
        n = int(next(it)); m = int(next(it))
        a = [int(next(it)) for _ in range(n)]
        st = RangeChminTree(a)
        for _ in range(m):
            op = int(next(it))
            if op == 0:
                l = int(next(it)); r = int(next(it)); x = int(next(it))
                st.chmin(l - 1, r - 1, x)
            elif op == 1:
                l = int(next(it)); r = int(next(it))
                out.append(str(st.query_max(l - 1, r - 1)))
            else:
                l = int(next(it)); r = int(next(it))
                out.append(str(st.query_sum(l - 1, r - 1)))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()
```
