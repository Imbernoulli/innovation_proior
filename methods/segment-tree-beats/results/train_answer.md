We need an online structure for an integer array $a[1..n]$ under three interleaved operations: a range cap $\texttt{chmin}(l,r,x)$ that sets $a[i] \leftarrow \min(a[i], x)$ for all $i \in [l,r]$, a range maximum query, and a range sum query. Both the array and the operation stream can be huge, so re-scanning a range per operation is too slow. The two read-only queries are already comfortable for a segment tree: store an aggregate per node and merge the two children's answers when a query splits. The hard part is the update. If the update were a range add, a fully covered node of length $k$ would change its sum by $k\cdot x$; if it were a range assign, its sum would become $k\cdot x$. Both are closed over the ordinary sum/max summary, so lazy propagation handles them with a single tag. A $\texttt{chmin}$ is different in kind: $\min(a_i, x)$ touches exactly the entries above $x$ and leaves the rest alone, and a node's sum and maximum do not reveal how many entries exceed $x$ or how far they fall. Two segments can share length, maximum, and sum yet respond differently to the same cap. So ordinary lazy propagation stalls precisely at the point where it would need distributional information the node never stored, and the only correct fallback — descending to every affected leaf and clamping it one by one — is linear per update.

The method is Segment Tree Beats. The idea is to enrich each node's summary by exactly the one extra fact a cap needs, so that most fully covered nodes can still be updated in constant time, and to descend only in the genuinely insufficient case — then pay for those descents amortized. Each node over its interval stores four quantities: the maximum $M$, the strict second maximum $m$ (the largest value strictly below $M$, taken as $-\infty$ when the interval holds a single distinct value), the count $cnt$ of entries equal to $M$, and the sum $S$. A cap by $x$ at a node then splits cleanly into three cases. If $M \le x$, nothing in the node exceeds the cap and we return immediately — this case reuses the maximum we already keep for the range-max query. If $m < x < M$, then every value strictly below the maximum is already at or below $x$, so the only entries above the cap are the $cnt$ copies of $M$; the entire node updates in constant time by
$$S \mathrel{+}= cnt\,(x - M), \qquad M \leftarrow x,$$
with $m$ and $cnt$ unchanged, and a pending cap left for the descendants. The strict inequality $x > m$ is load-bearing and not a convenience: if $x = m$, the old maximum entries collapse onto the already-present level $m$, so the new count of the maximum would be $cnt$ plus the number of old-$m$ entries, a quantity the summary does not record. Equality must therefore fall to the recursive case. That leaves the third case, $x \le m$: now at least two distinct levels — the maximum and the strict second maximum — sit at or above the cap boundary, the four-number summary is deliberately too coarse to know how they redistribute, so we push any pending cap down, recurse into both children, and rebuild the node from them.

The merge that rebuilds a parent must preserve the word "strict" in the strict second maximum, and this is where the three branches of $\texttt{\_pushup}$ come from. The parent maximum is the larger of the two child maxima. When the child maxima tie, both children supply maximum-valued entries, so their counts add and the parent's strict second is $\max$ of the two child strict-seconds. When the left maximum strictly wins, the parent maximum and count come from the left child, and the parent's strict second is $\max$ of the left child's own strict second and the *right child's maximum* — it is the right maximum, not the right strict second, that is the largest value sitting just below the parent maximum. The right-wins case is symmetric, and the sum is always the sum of the children. The lazy cap is safe for the same reason the constant-time case is exact: a tag is created only when capping a fully covered node in the strict middle case, so the cap value lies strictly above that node's strict second maximum. Pushed to a child, either the child's maximum is already $\le x$ and the cap is a no-op, or the child's maximum exceeds $x$, in which case that child holds the parent's old maximum level while all its other distinct values are at most the parent's strict second, hence strictly below the cap — so the very same constant-time apply is valid there. A tag is just the cap still owed to descendants, and overwriting an old pending cap by a smaller new one is exactly the composition $\min(\cdot, x_{\text{old}}) \circ \min(\cdot, x_{\text{new}})$ when $x_{\text{new}} \le x_{\text{old}}$.

What makes this fast is an amortized argument over the whole operation sequence rather than per update, because a single $\texttt{chmin}$ can still open several nodes. Define the potential $\Phi = \sum_u d(u)$, where $d(u)$ is the number of distinct values currently in node $u$'s interval. A node covering $k$ positions has at most $k$ distinct values, and each tree level covers the array once, so initially $\Phi = O(n\log n)$. Consider one extra descent — a fully covered node $u$ with $x \le m_u < M_u$, where the range already covers the node yet we still recurse. Because $x \le m_u$, the node has at least two distinct top levels; if $x = m_u$ the maximum level merges onto the second level, and if $x < m_u$ both top levels (and possibly more) collapse onto the single new value $x$. Either way, once the update inside $u$ completes, $d(u)$ has dropped by at least one. A cap never manufactures arbitrary new values — every resulting value is either an old value that was already $\le x$ or the single cap value $x$ — so for a fully covered node the distinct count cannot rise. Distinct counts can only increase at the $O(\log n)$ partially covered boundary nodes of an update, where capped and uncapped children are merged, and each such node gains at most the one new value $x$. Hence one operation raises $\Phi$ by at most $O(\log n)$; over $q$ operations the total injectable increase is $O(q\log n)$ on top of the $O(n\log n)$ initial reserve. Since each extra descent spends at least one unit of $\Phi$ and $\Phi \ge 0$ always, the extra descents total $O((n+q)\log n)$, and adding the ordinary segment-tree visits gives an overall amortized $O((n+q)\log n)$ for the entire sequence of updates and queries. The concrete summary therefore maps directly to the storage — $\texttt{slot\_a}$ is $M$, $\texttt{slot\_b}$ is $m$, $\texttt{slot\_c}$ is $cnt$, $\texttt{sm}$ is $S$, and $\texttt{tag}$ is the pending cap — and the update simply executes the three cases without ever weakening the strict middle inequality.

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
