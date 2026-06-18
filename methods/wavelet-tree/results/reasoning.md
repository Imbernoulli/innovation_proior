In front of me is a fixed array of $n$ integers and a pile of queries, each asking one of two things about a contiguous slice `a[l..r]`: what is the $k$-th smallest value in that slice, and how many values in that slice are $\le x$. Think $n$ and the number of queries both around $10^5$. The honest first move is to just do what the query says. For the $k$-th smallest: copy out `a[l..r]`, sort it, read off position $k$. For the count: scan the slice and tally how many entries are $\le x$. Both correct, and on one small query, fine.

Cost it. The slice can be the whole array, so one $k$-th-smallest query is a sort of up to $n$ elements, $O(n\log n)$, and one count query is an $O(n)$ scan. With $q$ of them that's $O(qn\log n)$ — around $10^{10}$ and then some. Dead. And the thing that makes it dead is that every query re-examines its slice from scratch: I sort the same overlapping stretches of the array again and again, throwing the work away each time. The slice is contiguous, the array is fixed — surely I can preprocess once and then answer each query by *navigating* a prebuilt structure instead of touching every element of the slice.

What do these two queries really have in common? Both are about *where a value sits in sorted order, restricted to a window of positions*. "$k$-th smallest in `a[l..r]`" is an order statistic of the slice; "how many in `a[l..r]` are $\le x$" is a rank of the value $x$ within the slice. Order statistic and rank are inverses of each other on a sorted multiset — if I can answer one of them efficiently for an arbitrary window, the other should come almost for free from the same machinery. So let me hunt for *one* structure that, for any window of positions, lets me count how many of its elements have value below some threshold. That single primitive — "count elements of `a[l..r]` whose value is $\le$ some bound" — is literally the rank query, and I'll see in a moment it also drives the order statistic.

First, the values themselves are a nuisance: they can be huge, negative, sparse. None of the queries care about the actual magnitudes, only the relative order. So compress: take the sorted distinct values, and replace each $a_i$ by its index in that sorted list — a dense code in $[0, \sigma)$ where $\sigma$ is the number of distinct values. Codes preserve order, "$\le x$" becomes "$\le$ the largest code whose value is $\le x$", and the final answer's code maps back to a value at the very end. Now I'm working with an array of small integers, each in $[0, \sigma)$.

So I want: count how many of the positions $l..r$ carry a code below a threshold. The threshold ranges over the whole value domain $[0, \sigma)$, and the window $[l, r]$ ranges over all of $[1, n]$. Two axes — value and position — and I'm being asked, repeatedly, a 2D dominance count: positions in a range, values below a bound. The trouble is the two axes pull against each other. If I sort by value I lose the position window; if I keep position order I have to filter by value at query time. I want a structure where I can *split on value* but still *track position*.

Let me think about the order-statistic query directly, because it suggests a shape. The $k$-th smallest of the slice — if I knew how the slice splits around the *median of the value domain*, I'd be halfway there. Concretely: pick the midpoint $m$ of the value range $[0, \sigma)$. Every element of the slice is either "low" (code $\le m$) or "high" (code $> m$). Suppose I could count, quickly, how many elements of `a[l..r]` are low — call it $c_{\text{low}}$. Then the $k$-th smallest is decided in one comparison: if $k \le c_{\text{low}}$, the answer is a low value, and it's the $k$-th smallest *among the low elements of the slice*; if $k > c_{\text{low}}$, the answer is a high value, and it's the $(k - c_{\text{low}})$-th smallest among the high elements of the slice. Either way I've halved the value range I'm searching, and I've turned the question into the *same* question — a $k'$-th smallest within a slice — but now restricted to either the low or the high elements.

That's a recursion on the value range. Bisect $[0, \sigma)$ at its midpoint; descend into the half that contains the answer; repeat. After $\log \sigma$ steps the value range is a single value and that's the answer. So at each level I need just two things: (a) count how many elements of the current slice fall in the low half, and (b) figure out what the slice *becomes* when I drop into that half. Part (a) is the count primitive I already said I needed. Part (b) is the one I have no tool for yet — so the structure has to be built around making it cheap.

The root of the structure owns the full value range $[0, \sigma)$ and the *whole array* — all $n$ elements, in their original left-to-right order. I bisect the value range at $m = \lfloor(0 + (\sigma-1))/2\rfloor$. Now I walk the array once and label each element with a single bit: $0$ if its code is $\le m$ (low half), $1$ if its code is $> m$ (high half). Then I form two child sequences by *stably partitioning*: pull out the low-half elements in the order they appear and hand them to the left child (which will own value range $[0, m]$); pull out the high-half elements, in order, and hand them to the right child (value range $[m+1, \sigma-1]$). Recurse on each child with its own narrower value range. Stop when a node's value range is a single code — that's a leaf, every element under it has that one value.

Stable is the load-bearing word. By keeping each half in its original relative order, I make the children themselves valid arrays-with-position, so the *same* construction applies recursively and the position structure is never scrambled within a half. Each level of this tree holds every element exactly once (an element goes to exactly one child per split), so a level is a permutation of the $n$ elements, and there are $O(\log \sigma)$ levels. That's $O(n \log \sigma)$ elements stored across the whole tree.

But I haven't yet solved part (b): when the query slice `a[l..r]` is at some node and I decide to descend into a child, *what positions does that slice occupy in the child?* The child only holds a subset of the node's elements (the ones that took the corresponding bit), and they've been compacted to the front in stable order. I can't afford to track individual slice elements down the tree — that would be $O(\text{slice length})$ again. I need the slice, which is a contiguous range $[l, r)$ at this node, to map to a *contiguous range* at the child, computable in $O(1)$.

Stare at the stable partition. At this node, suppose I read its elements left to right and, for each prefix, I knew how many of them carried bit $0$ (went left). Call that prefix count $z(i) = $ (number of $0$-bits among the first $i$ elements of this node's sequence). Because the left child receives the $0$-elements *in order*, the element sitting at node-position $p$ that happens to be a $0$ lands at left-child position exactly $z(p)$ — the number of $0$s strictly before it, since those are the only elements that precede it among the lefts, and order is preserved. Likewise a $1$-element at node-position $p$ lands at right-child position $p - z(p)$ — the number of $1$s strictly before it. So the position map into a child is just a prefix count of the bit array.

Now push a contiguous half-open range $[l, r)$ through it. The left child receives, in order, exactly the $0$-elements of $[l, r)$. The first node-position is $l$; the number of $0$s strictly before $l$ is $z(l)$, so the leftmost of them sits at left-position $z(l)$. The number of $0$s strictly before $r$ is $z(r)$, so the range of left-child positions occupied by the $0$-elements of $[l, r)$ is exactly $[z(l), z(r))$ — still half-open, still contiguous, and its length $z(r) - z(l)$ is precisely how many elements of the slice went low. The high half is the mirror: the number of $1$s before $l$ is $l - z(l)$, before $r$ is $r - z(r)$, so the $1$-elements of $[l, r)$ occupy right-child positions $[l - z(l), r - z(r))$. Both children get a clean contiguous sub-slice, computed from two lookups of the same prefix-count array. Map both endpoints through the *same* function and the half-open convention does the off-by-one bookkeeping for me — no $+1$/$-1$ fudge, because $[l, r)$ with $z$ applied at both ends stays a correct half-open range. (If I'd used closed ranges I'd be juggling "before $l$" versus "up to and including $r$" and inviting an off-by-one; half-open kills that.)

So the auxiliary thing I store at each node is exactly this prefix-count-of-zeros array, $z[0..\text{len}]$, with $z[0] = 0$ and $z[i+1] = z[i] + (1 - \text{bit}_i)$. I never even need to store the bits or the child sequences themselves after building — the prefix array is all the query traversal touches. Build it during the stable partition: as I sweep the node's sequence deciding left vs right, I accumulate the prefix of zeros into $z$ and simultaneously emit the two child sequences. One linear pass per node, total $O(n \log \sigma)$ to build.

Let me now assemble the $k$-th-smallest descent with this mapping and make sure the indices line up. Convert the $1$-based query `a[l..r]` to a half-open $[l-1, r)$ at the root. At a node with current half-open range $[\text{lo}, \text{hi})$: read $z_{\text{lo}} = z[\text{lo}]$, $z_{\text{hi}} = z[\text{hi}]$; the number of slice elements going low is $\text{num\_left} = z_{\text{hi}} - z_{\text{lo}}$. If $k \le \text{num\_left}$, the answer is low: descend to the left child with range $[z_{\text{lo}}, z_{\text{hi}})$ and the same $k$. Otherwise the answer is high: subtract the low count, $k \mathrel{-}= \text{num\_left}$, and descend right with range $[\text{lo} - z_{\text{lo}}, \text{hi} - z_{\text{hi}})$. Stop at the leaf, whose single value code is the answer; map it back through `vals`. Each step is $O(1)$ and there are $O(\log \sigma)$ of them, so $O(\log \sigma)$ per query. The recursion bottoms out correctly because $k$ is always kept in range: at the root $1 \le k \le \text{hi} - \text{lo}$, and whichever branch I take, the new $k$ stays between $1$ and the new range's length (going left, $k \le \text{num\_left} = z_{\text{hi}} - z_{\text{lo}}$, the left range's length; going right, $k$ becomes $k - \text{num\_left} \le (\text{hi}-\text{lo}) - \text{num\_left}$, the right range's length). So I never fall off an end.

Let me hand-trace to be sure I trust the mapping, because a flipped prefix here would silently corrupt everything. Take a small array of codes, say $[3,1,2,1,0]$ (so $\sigma = 4$, value range $[0,3]$), and ask for the $2$nd smallest of the whole array, $[l,r) = [0,5)$, $k=2$ — the sorted slice is $[0,1,1,2,3]$ so the answer should be code $1$. At the root, $m = 1$; bits (code $\le 1$ is low): $3\!\to\!1, 1\!\to\!0, 2\!\to\!1, 1\!\to\!0, 0\!\to\!0$, so the bit array is $[1,0,1,0,0]$ and $z = [0,0,1,1,2,3]$. $z_{\text{lo}} = z[0] = 0$, $z_{\text{hi}} = z[5] = 3$, $\text{num\_left} = 3$. $k=2 \le 3$, go left with range $[0,3)$; left child owns value range $[0,1]$ and its stable sequence is the $0$-elements in order: $[1,1,0]$. Now $m = 0$; bits (code $\le 0$ low): $1\!\to\!1, 1\!\to\!1, 0\!\to\!0$, bit array $[1,1,0]$, $z=[0,0,0,1]$. Range $[0,3)$: $z_{\text{lo}}=0$, $z_{\text{hi}}=z[3]=1$, $\text{num\_left}=1$. $k=2 > 1$, so go right, $k \mathrel{-}= 1 \Rightarrow k=1$, right range $[\text{lo}-z_{\text{lo}}, \text{hi}-z_{\text{hi}}) = [0-0, 3-1) = [0,2)$. Right child owns value range $[1,1]$ — a single code, a leaf. Answer code $= 1$. Correct, and the index bookkeeping survived two levels. Good.

Now the rank query — count of `a[l..r]` with value $\le x$ — using the same skeleton. First turn $x$ into a code threshold: the elements with value $\le x$ are exactly those whose code is in $[0, x_c)$, where $x_c$ is the number of distinct values $\le x$ (a binary search in `vals`). So I want to count how many elements of the slice have code in $[0, x_c)$. The set $[0, x_c)$ is a *prefix* of the value domain, which means as I bisect the value range it interacts with the bisection cleanly: at any node owning value range $[\text{vlo}, \text{vhi}]$, if the whole node sits below the threshold ($\text{vhi} < x_c$) then *every* element of the current sub-slice qualifies — just add the slice length and stop; if the whole node sits at or above the threshold ($x_c \le \text{vlo}$) then *none* qualify — return $0$; otherwise the threshold splits this node, so push the slice into both children using the same prefix mapping and sum the two answers. Map the slice into the left child as $[z_{\text{lo}}, z_{\text{hi}})$ and into the right as $[\text{lo}-z_{\text{lo}}, \text{hi}-z_{\text{hi}})$ — identical to before. Because the threshold boundary follows a single root-to-leaf path, at most two nodes per level are ever *split* (the rest are wholly-in or wholly-out and terminate immediately), so this recursion also touches $O(\log \sigma)$ nodes — $O(\log \sigma)$ per query. And it's exactly the inverse machinery of the order statistic, as I suspected at the start: both ride on "map a position-range into a child by a prefix count of the split bitmap."

Two loose ends. Empty arrays and degenerate value ranges: if every value is equal, $\sigma = 1$, the root's value range is a single code, it's a leaf, and both queries answer from it directly (the $k$-th smallest is that value; the rank is the slice length if $x$ is at least that value, else $0$). Duplicates in general are handled automatically — equal codes travel together through every split and pile up at a leaf, and counting by multiplicity is exactly what the prefix counts do. Negative and gigantic inputs are gone after compression. Nothing special is needed.

Let me write it. The structure is a binary tree stored in parallel arrays: per node, its value-range bounds `lo[node]`, `hi[node]`, child ids `lc[node]`, `rc[node]`, and the prefix-zeros array as its auxiliary. `build` recursively creates a node, and if its range is more than one code, bisects at the midpoint, builds the prefix-zeros array while stably splitting the sequence into low and high child sequences, and recurses. `kth` runs the iterative left/right descent on half-open ranges; `rank_leq` runs the threshold-prefix recursion. Values are compressed up front and the final code is mapped back through `vals`.

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

The whole chain: re-sorting each slice throws away work because the array is fixed, so I preprocess once; both queries are really "count slice elements below a value bound," so I bisect the value domain and at each level mark every element low/high with one bit, stably partitioning the low elements and the high elements into two children that stay valid position-ordered arrays; storing only the prefix count of those bits lets a contiguous slice $[l,r)$ map to a contiguous sub-slice of a child in $O(1)$ — left to $[z(l), z(r))$, right to $[l-z(l), r-z(r))$ — so the $k$-th smallest descends by comparing $k$ to the low count and the rank query sums child counts along the threshold prefix, each in $O(\log \sigma)$ per query on an $O(n\log\sigma)$ build.
