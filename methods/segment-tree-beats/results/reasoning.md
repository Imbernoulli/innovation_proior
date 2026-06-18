I need an online structure for an array with range chmin, range max, and range sum. The max query and the sum query fit a normal segment tree: store an aggregate in each node, split a query over the two children, merge the answers. The trouble is the update. If the update were range add, a fully covered node with length `k` would change its sum by `k*x`; if it were range assign, its sum would become `k*x`. Those operations are closed over the usual node aggregate. A chmin is different. Replacing every value by `min(value, x)` changes exactly the values above `x`, and a node's sum alone does not tell me how many such values there are or how far they fall. Two segments can have the same sum and length but respond differently to the same cap. So ordinary lazy propagation stops at the point where it needs distributional information that the node does not store.

The slow fallback is clear: descend to every affected leaf and clamp it. That is correct, but a single update can touch a whole interval, so it can be linear. I need a node summary that sometimes lets me handle a fully covered node in constant time. A chmin by `x` does nothing to values already at most `x`, so the most basic useful summary is the node maximum `M`. If `M <= x`, nothing in this node changes, and I can return without descending. That part is already available because range max is one of the required queries.

Now suppose `M > x`. Some values in the node change. I can still do the whole node at once if I know that the only values above `x` are the values equal to `M`. The way to test that is to store the largest value strictly below `M`; call it `m`, with `-infinity` when the whole node has one distinct value. If `m < x < M`, then everything below the maximum is already below the cap, and every value above the cap is exactly `M`. If I also store how many entries equal `M`, say `cnt`, the sum changes by `cnt * (x - M)`, the maximum becomes `x`, the strict second value stays `m`, and the count of the maximum stays `cnt`. The strict inequality matters. If `x == m`, the old maximum entries fall onto an already existing value, so the new maximum count would be `cnt` plus the number of old `m` entries, and I do not store that second count. Equality must therefore go to the recursive case.

That gives the real three-way update at a node. If `x >= M`, return. If the node is fully covered and `m < x < M`, apply the constant-time change and leave a pending cap for the children. If `x <= m`, at least two distinct value levels in this node are at or above the cap boundary: the maximum level and the strict second level. The summary `(M, m, cnt, sum)` is intentionally too small to know how all those levels are distributed, so I push any pending cap, recurse to the children, and rebuild this node from the children.

The merge rule has to preserve the word "strict" in strict second maximum. The parent maximum is the larger child maximum. If the child maxima are equal, both children contribute maximum-valued entries, so the counts add and the parent second value is `max(left_second, right_second)`. If the left child maximum is larger, the parent maximum and its count come from the left child, and the parent second value is the larger of the left child's own second value and the right child's maximum. The right child's maximum, not its second value, is the losing value just below the parent maximum. The symmetric rule holds when the right child maximum is larger. The sum is just the sum of child sums.

I also need to be sure the lazy cap is safe when it is eventually pushed down. A cap is created only in the constant-time case for a fully covered parent, so it lies strictly above the parent's strict second value. Consider a child. If the child's maximum is already at most the cap, applying the cap to that child is a no-op. If the child's maximum is above the cap, then that child must contain the parent's old maximum level, and every other distinct value inside the child is at most the parent's strict second value, hence strictly below the cap. So the same constant-time operation is valid on the child. The tag is simply the cap value still owed to descendants; if a later cap lowers the same node again, replacing the old pending cap by the smaller new maximum is exactly the composition of the two caps.

The recursion is the part that needs an amortized bound. A normal segment-tree operation touches `O(log n)` covered nodes and boundary nodes; the extra cost comes from a fully covered node where `x <= m`, because the range already covers the node yet I still descend. Define a potential `Phi = sum_u d(u)`, where `d(u)` is the number of distinct values currently present in node `u`'s interval. A node covering `k` array positions has at most `k` distinct values, and each tree level covers the array once, so initially `Phi <= O(n log n)`.

Look at one extra descent at a fully covered node `u`. Since `x <= m_u < M_u`, the node contains at least two distinct top levels. If `x == m_u`, the old maximum level falls onto the old second level; if `x < m_u`, both of those top levels, and maybe more levels, fall onto the single new value `x`. Either way, after this update has been completed inside `u`, the number of distinct values in `u`'s interval has dropped by at least one. A chmin cannot create arbitrary new values; every output value is either an old value that was already at most `x` or the single cap value `x`. For a fully covered node that means the distinct count never increases. Increases can only show up in ancestors of partially covered work, where changed and unchanged children are merged, and each such boundary node can gain at most the one new value `x`. There are only `O(log n)` boundary nodes for one interval update, so one operation can raise `Phi` by at most `O(log n)`. Across all operations, the total available increase is `O(q log n)`, and the initial reserve is `O(n log n)`. Since every extra descent spends at least one unit of this potential and `Phi` never becomes negative, the total number of extra descents is `O((n + q) log n)`. Adding the ordinary segment-tree visits gives amortized `O((n + q) log n)` total time for the whole sequence of updates and queries.

So the final node summary is exactly what the cap needs: `slot_a` is the maximum `M`, `slot_b` is the strict second maximum `m`, `slot_c` is the count of entries equal to `M`, and `sm` is the sum. The code follows the three cases without weakening the strict middle case.

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

The result is a segment tree whose cheap cases are exact, whose recursive case is exactly the one where the node summary is insufficient, and whose expensive descents are paid for by the drop in distinct values across the tree.
