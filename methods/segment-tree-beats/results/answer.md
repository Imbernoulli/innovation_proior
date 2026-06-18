# Segment Tree Beats for Range Chmin, Max, and Sum

Maintain a segment tree whose node summary is strong enough to sometimes apply a
range `chmin` to a fully covered node without opening it. Each node stores:

- `M`: the maximum value in the node interval;
- `m`: the strict second maximum, or `-infinity` if the interval has one distinct value;
- `cnt`: the number of entries equal to `M`;
- `S`: the interval sum.

For a chmin by `x`, the node has three cases. If `M <= x`, no value changes. If
`m < x < M`, the only values above `x` are the `cnt` copies of `M`, so the node is
updated in constant time: `S += cnt * (x - M)` and `M = x`. The strict inequality
is essential: when `x == m`, the old maximum falls onto an existing level and the
new maximum count is not known from this summary, so the update must recurse. If
`x <= m`, recurse into the children and recompute the node.

The merge rule is exact: when child maxima tie, add their counts and take the
larger child second maximum; when one child maximum wins, the losing child's
maximum is a candidate for the parent's strict second maximum. The amortized cost
is `O((n + q) log n)`: every extra descent past a fully covered node with
`x <= m` drops that node's distinct-value count by at least one after the update
inside it completes, while each update can introduce the new cap value only along
`O(log n)` partially covered boundary nodes.

```python
import sys
from sys import setrecursionlimit

NEG = float("-inf")


class RangeChminTree:
    def __init__(self, a):
        self.n = len(a)
        size = 4 * self.n
        self.slot_a = [0] * size      # maximum M
        self.slot_b = [NEG] * size    # strict second maximum m
        self.slot_c = [0] * size      # count of entries equal to M
        self.sm = [0] * size          # sum over the node interval
        self.tag = [NEG] * size       # pending cap, if any
        self._build(1, 0, self.n - 1, a)

    def _pushup(self, u):
        ls, rs = u << 1, u << 1 | 1
        self.sm[u] = self.sm[ls] + self.sm[rs]
        if self.slot_a[ls] == self.slot_a[rs]:
            self.slot_a[u] = self.slot_a[ls]
            self.slot_b[u] = max(self.slot_b[ls], self.slot_b[rs])
            self.slot_c[u] = self.slot_c[ls] + self.slot_c[rs]
        elif self.slot_a[ls] > self.slot_a[rs]:
            self.slot_a[u] = self.slot_a[ls]
            self.slot_b[u] = max(self.slot_b[ls], self.slot_a[rs])
            self.slot_c[u] = self.slot_c[ls]
        else:
            self.slot_a[u] = self.slot_a[rs]
            self.slot_b[u] = max(self.slot_a[ls], self.slot_b[rs])
            self.slot_c[u] = self.slot_c[rs]

    def _apply(self, u, x):
        if self.slot_a[u] <= x:
            return
        self.sm[u] += (x - self.slot_a[u]) * self.slot_c[u]
        self.slot_a[u] = x
        self.tag[u] = x

    def _pushdown(self, u):
        if self.tag[u] == NEG:
            return
        self._apply(u << 1, self.tag[u])
        self._apply(u << 1 | 1, self.tag[u])
        self.tag[u] = NEG

    def _build(self, u, l, r, a):
        if l == r:
            self.sm[u] = a[l]
            self.slot_a[u] = a[l]
            self.slot_b[u] = NEG
            self.slot_c[u] = 1
            return
        mid = (l + r) >> 1
        self._build(u << 1, l, mid, a)
        self._build(u << 1 | 1, mid + 1, r, a)
        self._pushup(u)

    def chmin(self, ql, qr, x, u=1, l=0, r=None):
        if r is None:
            r = self.n - 1
        if self.slot_a[u] <= x:
            return
        if ql <= l and r <= qr and self.slot_b[u] < x:
            self._apply(u, x)
            return
        mid = (l + r) >> 1
        self._pushdown(u)
        if ql <= mid:
            self.chmin(ql, qr, x, u << 1, l, mid)
        if mid < qr:
            self.chmin(ql, qr, x, u << 1 | 1, mid + 1, r)
        self._pushup(u)

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
        return self.slot_a[u]


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
