I start with the fixed array and the query I have to answer: in `a[l..r]`, return the `k`-th smallest value. If I copy that slice, sort it, and index into it, the answer is obviously right, including duplicates, but one wide query costs `O(n log n)`. With many wide queries, I am paying for the same overlapping elements again and again. I need the query to reuse preprocessing from the fixed array.

The order-statistic part has a natural shape if I stop thinking about positions and think about values. I can compress all distinct array values into indices `1..self.len`, then keep a segment tree over that value domain. A node stores how many inserted elements fall into its value interval. If I already had the exact multiset for one query range in that tree, the `k`-th smallest is just a descent: count the items in the left value half; if `k` is no larger than that count, keep looking left, otherwise skip those left-half items and look right with `k` reduced by that count. The leaf gives a compressed value index, and `self.vals[index - 1]` gives the real value. That is the right inner operation, but it only solves the problem once the tree contains exactly `a[l..r]`.

For a static array, a range is the difference between two prefixes. If `T_i` is a value-count tree containing the multiset `a[1..i]`, then the count of range elements inside any value interval is the count in `T_r` minus the count in `T_{l-1}`. This works because every `T_i` uses the same compressed value domain, so the node covering a value interval means the same thing in both trees. During the order-statistic descent I do not need to materialize the difference tree. At an internal node, the number of range elements in the left value half is `sum[left child in T_r] - sum[left child in T_{l-1}]`. If `k` fits in that number I descend into both left children; otherwise I descend into both right children and subtract that left count from `k`. The sign has to be `right prefix minus left-before-range prefix`, so the paired query should receive `u = root[l - 1]` and `v = root[r]`, then compute `self.sum[self.ls[v]] - self.sum[self.ls[u]]`.

The prefix idea gives the query formula, but the direct storage is impossible. There are `n + 1` prefix trees, and a full value tree has `O(self.len)` nodes, so independent copies would take `O(n * self.len)` memory. Since `self.len` can be `n`, that is quadratic. The consecutive prefixes expose the way out. Moving from `T_{i-1}` to `T_i` inserts only `a[i]`. In a value-count segment tree, one leaf increases by one, and that change affects only the ancestors of that leaf. Every subtree off that root-to-leaf path is identical before and after the insert.

So the new tree should own only the changed path and share everything else with the old tree. I cannot do that with the usual implicit heap layout where children are `2 * node` and `2 * node + 1`, because identity is tied to array position. I need explicit node ids and child links. The arrays `self.sum`, `self.ls`, and `self.rs` make a node pool: a node id points to its stored count and its two child ids. To insert into an old node `prev`, I allocate a fresh node `cur`, set its count to `self.sum[prev] + 1`, and initially copy both child links from `prev`. Then I recurse only into the half that contains the inserted compressed value, replacing that one child link with the freshly returned child. The untouched child still points to the old subtree. Since every write goes into a newly allocated node, older roots remain valid.

That gives one root per prefix: `root[0]` is the empty tree, and `root[i]` is the root returned after inserting `a[i-1]` into `root[i-1]`. I do not need to build a complete zero tree. Node `0` can stand for every absent zero-count subtree: `self.sum[0]`, `self.ls[0]`, and `self.rs[0]` are all zero, so following missing links reads zeros consistently. Each inserted value creates one node per level plus the leaf, hence `O(log self.len)` new nodes per array element and `O(n log n)` total nodes.

I want to check the query walk on a concrete case before writing it. Let `a = [5, 1, 3]`, with compressed values `[1, 3, 5]`, so `5 -> 3`, `1 -> 1`, and `3 -> 2`. For the query `k = 2` in `a[1..3]`, I compare `root[0]` and `root[3]`; the multiset is all three indices, so the descent reaches compressed index `2`, value `3`. For the query `k = 1` in `a[2..3]`, I compare `root[1]` and `root[3]`. In the left half `[1, 2]`, `root[3]` has two elements and `root[1]` has none, so I go left. Inside `[1, 2]`, the left leaf has difference one, so I return compressed index `1`, value `1`. That matches sorting `[1, 3]`.

The implementation follows those invariants directly: `_insert` returns a new root after copying one path; `build` stores one root for each prefix; `_query` walks two roots in lockstep and uses the difference of their left counts; `kth` maps the reached compressed index back to the original value.

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
