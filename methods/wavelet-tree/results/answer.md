# Wavelet tree (value-bisection + per-level rank bitmaps)

## Problem

For a static array of $n$ integers, answer two kinds of query on a window of
positions `a[l..r]` (1-based): the $k$-th smallest value in the window, and the
number of window elements that are $\le x$ (range rank). Values may be negative and
may repeat; $n, q$ up to $\sim 10^5$.

## Key idea

**Recursively bisect the value range, not the positions.** Compress values to dense
codes in $[0, \sigma)$. The root owns the whole code range $[0, \sigma)$ and the whole
array in original order. Bisect at $m = \lfloor(\text{lo}+\text{hi})/2\rfloor$: mark
each element with a bit — $0$ if its code $\le m$ (low half), $1$ if $> m$ (high
half) — and **stably partition** the sequence so the low (bit-0) elements, in order,
form the left child (range $[\text{lo}, m]$) and the high elements form the right
child (range $[m+1, \text{hi}]$). Recurse; a node whose range is a single code is a
leaf. Each level is a permutation of all $n$ elements, depth is $O(\log \sigma)$.

**Store only the prefix-rank of the split bitmap.** At each node keep
$z[0..\text{len}]$ with $z[i]$ = number of bit-$0$ (low) elements among the first $i$
elements of the node's sequence. Because the stable partition preserves order, the
prefix count is exactly the position map into the children: an element at node
position $p$ that is low lands at left position $z(p)$; if high, at right position
$p - z(p)$. So a contiguous half-open slice $[l, r)$ of a node maps to a contiguous
slice of each child:

$$\text{left: } [\,z(l),\ z(r)\,), \qquad \text{right: } [\,l - z(l),\ r - z(r)\,).$$

Both endpoints go through the *same* prefix function; the half-open convention
handles the off-by-one with no $\pm1$ correction. The left slice's length
$z(r) - z(l)$ is exactly how many slice elements fell in the low half.

**$k$-th smallest** descends from the root on half-open ranges. At a node with slice
$[\text{lo}, \text{hi})$, let $\text{num\_left} = z(\text{hi}) - z(\text{lo})$. If
$k \le \text{num\_left}$ the answer is in the low half: go left with range
$[z(\text{lo}), z(\text{hi}))$, same $k$. Otherwise it is in the high half: set
$k \mathrel{-}= \text{num\_left}$ and go right with range
$[\text{lo} - z(\text{lo}), \text{hi} - z(\text{hi}))$. The leaf's single code is the
answer; map it back to a value. $O(\log \sigma)$.

**Range rank** (count $\le x$) reuses the same mapping. With $x_c$ = number of codes
$\le x$, the qualifying elements are those with code in $[0, x_c)$ — a prefix of the
value domain. Recurse from the root: if a node's range is wholly below the threshold
($\text{hi} < x_c$) add the whole slice length; if wholly at/above it
($x_c \le \text{lo}$) add $0$; otherwise split the slice into both children by the
prefix map and sum. The threshold follows one root-to-leaf path, so at most two nodes
per level are split — $O(\log \sigma)$.

## Algorithm

1. Compress values to dense codes in $[0, \sigma)$ via the sorted distinct values.
2. Build by value-bisection: at each node bisect the code range, scan the sequence to
   build the prefix-zeros array while stably splitting into the low/high child
   sequences; recurse to leaves.
3. `kth(l, r, k)`: iterative left/right descent comparing $k$ to the low count.
4. `rank_leq(l, r, x)`: threshold-prefix recursion summing child counts.

## Code

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
        codes = [self.code(x) for x in self.a]

        def rec(seq, vlo, vhi):
            node = self._new(vlo, vhi)
            if vlo == vhi:                     # leaf: one value, no split
                self.aux[node] = [0] * (len(seq) + 1)
                return node
            mid = (vlo + vhi) >> 1             # bisect the value range
            z = [0] * (len(seq) + 1)           # prefix count of low elements
            left, right = [], []              # stable partition into the two halves
            for i, c in enumerate(seq):
                if c <= mid:                  # low half -> bit 0, goes left
                    z[i + 1] = z[i] + 1
                    left.append(c)
                else:                         # high half -> bit 1, goes right
                    z[i + 1] = z[i]
                    right.append(c)
            self.aux[node] = z
            self.lc[node] = rec(left, vlo, mid)
            self.rc[node] = rec(right, mid + 1, vhi)
            return node

        if self.n == 0:
            self.root = self._new(0, 0)
            self.aux[self.root] = [0]
            return
        self.root = rec(codes, 0, self.sigma - 1)

    def kth(self, l, r, k):
        """k-th smallest value in a[l..r] (1-based l, r; 1-based k)."""
        lo, hi = l - 1, r                      # half-open slice [lo, hi)
        node = self.root
        while self.lo[node] != self.hi[node]:  # until a single-value leaf
            z = self.aux[node]
            zl, zr = z[lo], z[hi]
            num_left = zr - zl                 # slice elements in the low half
            if k <= num_left:                  # answer is low: map slice into left
                lo, hi = zl, zr
                node = self.lc[node]
            else:                              # answer is high: drop the low count
                k -= num_left
                lo, hi = lo - zl, hi - zr      # map slice into right
                node = self.rc[node]
        return self.vals[self.lo[node]]

    def rank_leq(self, l, r, x):
        """Number of elements in a[l..r] with value <= x (1-based l, r)."""
        xc = bisect_right(self.vals, x)        # codes <= x are exactly [0, xc)

        def rec(node, lo, hi):
            if hi <= lo or xc <= self.lo[node]:   # empty slice / node wholly above x
                return 0
            if self.hi[node] < xc:                # node wholly at or below x
                return hi - lo
            z = self.aux[node]
            zl, zr = z[lo], z[hi]                 # threshold splits this node
            return rec(self.lc[node], zl, zr) + rec(self.rc[node], lo - zl, hi - zr)

        return rec(self.root, l - 1, r)


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

## Complexity

- **Build:** $O(n \log \sigma)$ time and memory — $O(\log \sigma)$ levels, each a
  linear pass over a permutation of all $n$ elements; per node a prefix-zeros array.
- **Query:** $O(\log \sigma)$ per `kth` (one root-to-leaf descent) and per `rank_leq`
  (the threshold splits at most two nodes per level). With value compression,
  $\sigma \le n$, so both are $O(\log n)$.
