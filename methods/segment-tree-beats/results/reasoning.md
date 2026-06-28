I need an online structure for an array with range chmin, range max, and range sum. The max query and the sum query fit a normal segment tree: store an aggregate in each node, split a query over the two children, merge the answers. The trouble is the update. If the update were range add, a fully covered node with length `k` would change its sum by `k*x`; if it were range assign, its sum would become `k*x`. Those operations are closed over the usual node aggregate. A chmin is different. Replacing every value by `min(value, x)` changes exactly the values above `x`, and a node's sum alone does not tell me how many such values there are or how far they fall. To see this concretely: the segments `[5, 5, 0]` and `[4, 3, 3]` both have sum `10` and length `3`, but a chmin by `4` sends the first to `[4, 4, 0]` (sum `8`) and leaves the second untouched (sum `10`). Same sum and length in, different sum out. So ordinary lazy propagation stops at the point where it needs distributional information that the node does not store.

The slow fallback is clear: descend to every affected leaf and clamp it. That is correct, but a single update can touch a whole interval, so it can be linear. I need a node summary that sometimes lets me handle a fully covered node in constant time. A chmin by `x` does nothing to values already at most `x`, so the most basic useful summary is the node maximum `M`. If `M <= x`, nothing in this node changes, and I can return without descending. That part is already available because range max is one of the required queries.

Now suppose `M > x`. Some values in the node change. I can still hope to do the whole node at once, but only if the cap interacts with a single value level. The clean situation is when the only values above `x` are the values equal to `M`. To test that, I want to know the largest value strictly below `M`. Call it `m`, with `-infinity` when the whole node has one distinct value. If `m < x < M`, then everything below the maximum is already at or below the cap, and every value above the cap is exactly `M`. Storing how many entries equal `M`, say `cnt`, then lets me update the sum directly: each of the `cnt` copies drops from `M` to `x`, so `S` changes by `cnt * (x - M)`, the maximum becomes `x`, and the count of the maximum stays `cnt` because those entries are still tied at the new top. The strict second value `m` is untouched.

Let me check that arithmetic on a real node before trusting it. Take the array `[5, 5, 3, 8]`. Built as a tree its root has `M = 8`, `m = 5`, `cnt = 1`, `S = 21` — I can read those off by hand: the maximum is `8` with one copy, the next distinct value down is `5`, and `5+5+3+8 = 21`. Now apply chmin by `x = 4` over the whole range. Here `m = 5` is not below `4`, so this is *not* the constant-time case for the root; but it lets me sanity-check the end result against brute force. Clamping by hand gives `[4, 4, 3, 4]`, sum `15`, max `4`. Running the structure on the same input returns sum `15` and max `4`. They agree, which at least tells me the recursion and the summary stay consistent on a case where the fast rule does *not* fire.

The strict inequality `m < x` is the part I am least sure of, so I want to pin down what breaks when it is violated. Suppose `x == m`. The old maximum entries fall onto the value `m`, which already exists in the node. The new maximum is then `m = x`, and its count is the old `cnt` plus the number of entries that were already equal to `m` — a quantity I deliberately do not store. So with `x == m` the constant-time rule would set the right new maximum but the wrong count, and every later sum update that multiplies by `cnt` would be off. Equality must therefore go to the recursive case, alongside `x < m`. That fixes the fast-path guard as exactly `m < x < M`.

So the node update has three branches. If `x >= M`, return. If the node is fully covered and `m < x < M`, apply the constant-time change and leave a pending cap for the children. If `x <= m`, at least two distinct value levels in this node sit at or above the cap boundary — the maximum level and the strict second level — and the summary `(M, m, cnt, S)` is too small to know how those and any lower levels are distributed, so I push any pending cap, recurse to the children, and rebuild this node from the children.

Rebuilding is the merge, and it has to preserve the word "strict" in strict second maximum, which is the easiest thing to get wrong. The parent maximum is the larger child maximum; the sum is the sum of child sums. The delicate field is `m`. If the child maxima are equal, both children contribute maximum-valued entries, so the counts add and the parent strict second is `max(left.m, right.m)`. If one child maximum strictly wins — say the left — the parent maximum and count come from the left, and the parent strict second is the larger of the left child's *own* second value and the right child's *maximum*. The point I want to be careful about is that it is the right child's maximum, not its second value, that is the candidate: the right child's whole top level loses to the left maximum and is therefore the largest thing strictly below the parent maximum that the right side offers.

I do not trust that phrasing until I see it produce the right answer on a tree where a child loses. Take `a = [5, 5, 8, 3]`, built over `[0..1]` and `[2..3]`. The right node holds `[8, 3]`: maxima `8` vs `3`, left wins, so by my rule `M = 8`, `cnt = 1`, and `m = max(left.m, right.max) = max(-inf, 3) = 3`. That `3` is exactly the losing value, and it is the right node's *maximum*, confirming the "max not second" wording is the operative one. The left node `[5, 5]` ties, giving `M = 5`, `m = -inf`, `cnt = 2`. At the root, left max `5` vs right max `8`, right wins, so `M = 8`, `cnt = 1`, and `m = max(left.max, right.m) = max(5, 3) = 5`. By hand the array's true second-largest distinct value is `5`, so the root `m = 5` is correct. I coded these recurrences exactly as stated and read the node summaries back: root `(M=8, m=5, cnt=1, S=21)`, right child `(M=8, m=3, cnt=1)`, left child `(M=5, m=-inf, cnt=2)` — matching the hand computation in every field, including the two places where a child's maximum (not its second) supplies the parent's `m`.

I also need the lazy cap to be safe when it is eventually pushed down. A cap is created only in the constant-time case for a fully covered parent, so it lies strictly above the parent's strict second value. Consider a child. If the child's maximum is already at most the cap, applying the cap to that child is a no-op — and `_apply` guards on exactly `M <= x` returning immediately. If the child's maximum is above the cap, then that child must contain the parent's old maximum level, and every other distinct value inside the child is at most the parent's strict second value, hence strictly below the cap; so the child is itself in its own `m < x < M` situation and the same constant-time operation is valid one level down. The tag is just the cap value still owed to descendants; if a later, smaller cap lowers the same node again, replacing the old pending cap by the smaller new maximum is the composition of the two caps. I would not stake the whole structure on this argument alone, so I ran the finished code against a brute-force clamp on a few thousand random small instances mixing all three operations; every output matched. That does not prove the lazy invariant in general, but a tag bug almost always shows up on small random cases, and none did.

The recursion is the part that needs an amortized bound. A normal segment-tree operation touches `O(log n)` covered and boundary nodes; the extra cost comes from a fully covered node where `x <= m`, because the range already covers the node yet I still descend. To pay for those descents, define a potential `Phi = sum_u d(u)`, where `d(u)` is the number of distinct values currently present in node `u`'s interval. A node covering `k` array positions has at most `k` distinct values, and each tree level covers the array once, so initially `Phi <= O(n log n)`.

Look at one extra descent at a fully covered node `u`. Since `x <= m_u < M_u`, the node contains at least two distinct top levels. I want to claim that completing the update inside `u` strictly lowers `d(u)`. Concretely, take `u`'s values `[9, 9, 7, 2, 2]`, so `d(u) = 3` over `{9, 7, 2}`, with `M = 9`, `m = 7`. A cap with `x < m`, say `x = 5`: the result is `[5, 5, 5, 2, 2]`, distinct set `{5, 2}`, count `2` — down by one, because the two top levels `9` and `7` both collapse onto the single new value `5`. The boundary subcase `x == m = 7`: the result is `[7, 7, 7, 2, 2]`, distinct `{7, 2}`, count `2` — again down by one, because the `9` level merges into the existing `7` level. So in both subcases of `x <= m`, `d(u)` drops by at least one. A chmin cannot create arbitrary new values either: every output value is either an old value already at most `x` or the single cap value `x`, so for a fully covered node the distinct count can only fall. Distinct counts *can* rise, but only at partially covered boundary nodes, where a changed child and an unchanged child get merged and the parent can gain the one new value `x`; there are `O(log n)` such boundary nodes per update, so one operation raises `Phi` by at most `O(log n)`. Across all operations the total available increase is `O(q log n)` on top of the initial reserve `O(n log n)`. Each extra descent spends at least one unit of `Phi`, and `Phi` never goes negative, so the number of extra descents over the whole run is `O((n + q) log n)`. Adding the ordinary segment-tree visits gives amortized `O((n + q) log n)` for the whole sequence.

That settles the node summary: `slot_a` is the maximum `M`, `slot_b` is the strict second maximum `m`, `slot_c` is the count of entries equal to `M`, and `sm` is the sum. The update follows the three cases with the fast path guarded by the strict middle case `m < x < M`, and the merge writes `m` from the losing child's maximum.

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
