# Context

## Problem

Given a static array of $n$ integers, answer queries: the $k$-th smallest value in
`a[l..r]`, and the number of elements in `a[l..r]` that are $\le x$ (range rank).
($n, q$ up to $\sim 10^5$.)

The array is fixed after input; queries do not update it. Indices are $1$-based with
$1 \le l \le r \le n$, and for the order-statistic query $k$ is $1$-based with
$1 \le k \le r - l + 1$. Values may be negative and may repeat; equal values count
with multiplicity. Sorting `a[l..r]` per query, or scanning the slice to count
elements $\le x$, is correct but costs $O((r-l)\log(r-l))$ or $O(r-l)$ per query —
too slow when both $n$ and $q$ are large.

## Code framework

The scaffold fixes the input/output shape, stored fields, and public query methods.
`build`, `kth`, and `rank_leq` are the slots to fill.

```python
import sys
from bisect import bisect_left, bisect_right


class RangeQuery:
    def __init__(self, a):
        self.a = a
        self.n = len(a)
        self.vals = sorted(set(a))
        self.sigma = len(self.vals)
        self.lo = []
        self.hi = []
        self.lc = []
        self.rc = []
        self.aux = []
        self.root = -1
        self.build()

    def code(self, x):
        return bisect_left(self.vals, x)

    def _new(self, vlo, vhi):
        self.lo.append(vlo)
        self.hi.append(vhi)
        self.lc.append(-1)
        self.rc.append(-1)
        self.aux.append(None)
        return len(self.lo) - 1

    def build(self):
        # TODO
        pass

    def kth(self, l, r, k):
        """k-th smallest value in a[l..r] (1-based l, r; 1-based k)."""
        # TODO
        pass

    def rank_leq(self, l, r, x):
        """Number of elements in a[l..r] with value <= x (1-based l, r)."""
        # TODO
        pass


def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it)); q = int(next(it))
    a = [int(next(it)) for _ in range(n)]
    w = RangeQuery(a)
    out = []
    for _ in range(q):
        t = int(next(it))
        if t == 1:
            l = int(next(it)); r = int(next(it)); k = int(next(it))
            out.append(str(w.kth(l, r, k)))
        else:
            l = int(next(it)); r = int(next(it)); x = int(next(it))
            out.append(str(w.rank_leq(l, r, x)))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()
```
