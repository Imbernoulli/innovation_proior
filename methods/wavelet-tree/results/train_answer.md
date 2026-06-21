I have a fixed array of $n$ integers and a stream of $q$ queries, each asking one of two things about a contiguous window `a[l..r]` (1-based): the $k$-th smallest value in the window, or how many of its elements are $\le x$ (a range rank). Values may be negative, may repeat, and $n, q$ are both around $10^5$. The honest brute force is to do exactly what each query says — copy out the slice and sort it to read off position $k$, or scan the slice and tally the elements $\le x$. Both are correct, but the slice can be the entire array, so a single order-statistic query is an $O(n \log n)$ sort and a single count is an $O(n)$ scan, giving $O(q\,n\log n)$ overall — around $10^{10}$, hopelessly slow. The reason it is slow is structural: every query re-examines its slice from scratch, sorting the same overlapping stretches again and again and discarding the work. But the array is fixed and the slice is contiguous, so the right move is to preprocess once and answer each query by *navigating* a prebuilt structure rather than touching every element of the window. The naive 2D view — index the elements by both position and value and answer a dominance count (positions in a range, values below a bound) — exposes the real obstacle: the two axes pull against each other, because sorting by value destroys the position window and keeping position order forces value filtering at query time. What I need is a structure that splits on value while still tracking position.

I propose the wavelet tree, built by value-bisection with per-level rank bitmaps. The first observation is that both queries share one primitive: counting how many elements of a window carry a value below some threshold. That count *is* the rank query, and it also drives the order statistic, since order statistic and rank are inverses on a sorted multiset. The values themselves are a nuisance — huge, negative, sparse — and none of the queries care about magnitude, only relative order, so I compress: take the sorted distinct values and replace each $a_i$ by its index in that list, a dense code in $[0, \sigma)$ where $\sigma$ is the number of distinct values. Codes preserve order, "$\le x$" becomes "code in $[0, x_c)$" for a binary-searched threshold $x_c$, and the answer's code maps back to a value at the end.

The structure is a binary tree over the *value* range. The root owns the full code range $[0, \sigma)$ and the whole array in original left-to-right order. I bisect the value range at its midpoint $m = \lfloor(\text{lo}+\text{hi})/2\rfloor$, walk the node's sequence once, and label each element with a single bit: $0$ if its code $\le m$ (low half, goes left) and $1$ if its code $> m$ (high half, goes right). Then I form the two children by a *stable* partition — the low elements, in their original relative order, become the left child's sequence (owning value range $[\text{lo}, m]$), and the high elements, in order, become the right child's (owning $[m+1, \text{hi}]$). Recurse on each child with its narrower range; a node whose range is a single code is a leaf, under which every element shares that one value. Stability is load-bearing: by keeping each half in original order, the children are themselves valid position-ordered arrays, so the same construction recurses cleanly and position structure is never scrambled within a half. Each level holds every element exactly once (each element takes exactly one bit per split), so a level is a permutation of all $n$ elements and there are $O(\log \sigma)$ levels — $O(n\log\sigma)$ stored.

The crux is mapping a query window down the tree without ever tracking individual elements, which would cost $O(\text{slice length})$ again. The window is a contiguous range at a node, and I need it to map to a contiguous range at the child in $O(1)$. The stable partition makes this exact. Let $z(i)$ be the number of bit-$0$ (low) elements among the first $i$ elements of the node's sequence — a prefix count of zeros, with $z[0]=0$ and $z[i+1] = z[i] + (1-\text{bit}_i)$. Because the left child receives the $0$-elements in order, an element at node position $p$ that is low lands at left-child position exactly $z(p)$ (the count of lows strictly before it); a high element at position $p$ lands at right-child position $p - z(p)$ (the count of highs strictly before it). Pushing a half-open range $[l, r)$ through this, the $0$-elements of $[l,r)$ occupy left-child positions $[z(l), z(r))$ and the $1$-elements occupy right-child positions $[l - z(l), r - z(r))$:

$$\text{left: } [\,z(l),\ z(r)\,), \qquad \text{right: } [\,l - z(l),\ r - z(r)\,).$$

Both endpoints go through the *same* prefix function, and the half-open convention absorbs the off-by-one with no $\pm 1$ correction — closed ranges would force juggling "before $l$" against "up to and including $r$" and invite an off-by-one. The left slice's length $z(r)-z(l)$ is precisely how many window elements fell in the low half. So the only thing I store per node is this prefix-zeros array $z$; I never need the bits or child sequences themselves at query time, and I build $z$ in the same linear sweep that emits the two child sequences — one pass per node, $O(n\log\sigma)$ total.

With this map, the $k$-th smallest is a single root-to-leaf descent on half-open ranges. Convert the 1-based window to $[l-1, r)$ at the root. At a node with current range $[\text{lo}, \text{hi})$, let $\text{num\_left} = z(\text{hi}) - z(\text{lo})$ be the count of window elements in the low half. If $k \le \text{num\_left}$, the answer is a low value: descend left with range $[z(\text{lo}), z(\text{hi}))$ keeping the same $k$. Otherwise it is a high value: subtract the low count, $k \mathrel{-}= \text{num\_left}$, and descend right with range $[\text{lo}-z(\text{lo}), \text{hi}-z(\text{hi}))$. The leaf's single code is the answer, mapped back through the value list. Each step halves the value range and is $O(1)$, so the query is $O(\log\sigma)$, and $k$ stays in range at every step — going left $k \le \text{num\_left}$, the left range's length; going right $k - \text{num\_left}$ stays within the right range's length — so the descent never falls off an end.

The range rank reuses the identical mapping. The qualifying elements are those with code in $[0, x_c)$, a prefix of the value domain, where $x_c$ is the number of distinct values $\le x$. Recurse from the root: at a node owning value range $[\text{vlo}, \text{vhi}]$, if the node sits wholly below the threshold ($\text{vhi} < x_c$) then every element of the current sub-slice qualifies — add the slice length; if it sits wholly at or above it ($x_c \le \text{vlo}$) then none do — add $0$; otherwise the threshold splits the node, so map the slice into both children by the same prefix formula and sum. Because the threshold boundary follows a single root-to-leaf path, at most two nodes per level are ever split — the rest terminate immediately — so this too touches $O(\log\sigma)$ nodes. It is exactly the inverse machinery of the order statistic, as the shared primitive promised. Duplicates need no special handling: equal codes travel together through every split, pile up at a leaf, and the prefix counts tally them by multiplicity; a degenerate all-equal array has $\sigma=1$, a single leaf, and both queries answer directly from it. Build is $O(n\log\sigma)$ time and memory, each query is $O(\log\sigma)$, and with compression $\sigma \le n$ so both are $O(\log n)$.

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
