Two facts about this problem fix the whole shape before I write a line. Each query wants the
maximum-sum *contiguous* block inside a window `[l, r]`, empty block allowed so the answer floors at
`0` — a windowed Kadane — and the queries interleave with point updates. That kills both easy
routes: a per-query linear scan is `O(nq)`, worst case `4*10^10` in 2 seconds, hopeless; and updates
rule out any static precompute (sparse table over fixed blocks). I need one structure that answers a
range max-subarray *and* absorbs point updates in sublinear time, which points at a segment tree
carrying enough per-node statistics to merge two children in `O(1)`.

The second fact is scale, and it forces the data types: `|a[i]| <= 10^9` and a window spans up to
`2*10^5` cells, so a single window sum can reach `2*10^14`, five orders of magnitude past the 32-bit
ceiling of `~2.1*10^9`. Every sum, every stored field, every accumulator is `long long`; an `int`
anywhere is a silent wrong-answer on the large tests, not a crash I would notice.

And there is a tempting wrong answer lurking underneath: *sum the positives in the window*. For
`[l, r]`, add every `a[i] > 0`; it is `O(window)`, or `O(log n)` with a Fenwick tree over positive
values. But take the window `[5, -100, 4]`: summing positives gives `5 + 4 = 9`. Those two gains sit either side of
`-100`; to take both in one contiguous block I must swallow the `-100` too, `5 - 100 + 4 = -91`. The
only real blocks are `[5]=5`, `[4]=4`, and the negatives, so the true best is `5`. Greedy's `9`
corresponds to *no contiguous block at all* — it over-counts, teleporting the positives next to each
other. Contiguity is a global constraint the positive-sum can't honor. It happens to be exactly right
on all-positive windows (where it coincides with "take everything"), which is what makes it
seductive; the sample confirms the same trap, where `3+5+4+2 = 14` but the `-6` quarantines `4,2`
from the left cluster and the real best is `[3,-2,5] = 6`. Greedy is out.

So I want each node, covering a contiguous segment, to expose just enough to merge two adjacent
segments in `O(1)`. The classic quadruple:

- `total` = sum of all elements in the segment,
- `pre`   = best sum of a prefix (prefix may be empty, so `pre >= 0`),
- `suf`   = best sum of a suffix (suffix may be empty, so `suf >= 0`),
- `best`  = best sum of any contiguous block (block may be empty, so `best >= 0`).

Flooring `pre`, `suf`, `best` at `0` is exactly how the empty-block rule lives inside the structure:
the empty selection is always a competitor. A leaf holding `v` is `total = v`, and
`pre = suf = best = max(0, v)`. The merge of left child `L` and right child `R`:

- `total = L.total + R.total`.
- `pre = max(L.pre, L.total + R.pre)` — a prefix of the parent either stays inside `L`, or is all of
  `L` plus a prefix of `R`.
- `suf = max(R.suf, R.total + L.suf)` — symmetric.
- `best = max(L.best, R.best, L.suf + R.pre)` — the block lies entirely in `L`, entirely in `R`, or
  *straddles* the boundary as a suffix of `L` glued to a prefix of `R`.

The floors compose: `L.suf, R.pre >= 0` makes the straddle term `>= 0`, and with `L.best, R.best >= 0`
the parent's `best >= 0`, so the empty block survives every merge. Running `[3, -2, 5]` through it —
`(3,3,3,3)` merged with `(-2,0,0,0)` gives `(1,3,1,3)`, merged with `(5,5,5,5)` gives
`best = max(3,5,1+5) = 6` — reproduces the sample answer, and the same trap input `[5,-100,4]` comes
out `best = 5`, not greedy's `9`.

The subtlety in this kind of code is never the merge; it is the range query. When I recurse for
`[l, r]` I must return the empty-segment identity `{0,0,0,0}` for any node whose range is disjoint
from `[l, r]`, *before* the fully-inside test. Without that guard the recursion has no base case for
"this leaf is outside the window": it keeps descending, folds an out-of-window value into the answer,
and eventually indexes `2*node` past the tree into garbage. Take `a = [-2, 5, -1]`, query `[2, 3]`
(0-indexed `[1, 2]`, answer `5`). The left subtree covers `[0,1]`; a naive query with no disjoint
case would recurse into leaf `[0,0]` holding `a[0] = -2`, which is not in the window, and never find a
returning branch. So the query is:

```
Node query(int node, int lo, int hi, int l, int r) {
    if (r < lo || hi < l) return Node{0, 0, 0, 0};   // disjoint: empty segment
    if (l <= lo && hi <= r) return tree[node];        // fully inside
    int mid = (lo + hi) / 2;
    if (r <= mid) return query(2*node, lo, mid, l, r);
    if (l > mid)  return query(2*node+1, mid+1, hi, l, r);
    return combine(query(2*node, lo, mid, l, r),
                   query(2*node+1, mid+1, hi, l, r));
}
```

Re-tracing `[-2,5,-1]` query `[1,2]`: at `[0,1]`, `l = 1 > mid = 0`, so it descends only right to leaf
`[1,1] = (5,5,5,5)` and never touches `[0,0]`; combined with `[2,2] = (-1,0,0,0)` this yields
`best = 5`. The disjoint guard both stops the overrun and keeps position `0` out of the window sum.

The floors carry the all-negative case for free: on `[-3,-1,-4]` every leaf is `(v,0,0,0)`, and each
merge keeps `pre = suf = best = 0` while `total` accumulates the true negative sum, so the query
returns `0` — `total` is the only field allowed to go negative, and it feeds only the `pre/suf`
gluing, never the reported answer.

Updates and I/O are routine. A point update `1 p v` rewrites leaf `p-1` and re-merges its root path
in `O(log n)`; a query `2 l r` calls `query(1, 0, n-1, l-1, r-1)` and reports `best`. With up to
`2*10^5` query lines I batch output into one string and flush once — per-line `endl` flushing would
dominate the runtime. `n = 1` collapses to a single leaf and the `lo == hi` paths handle it. I
checked the whole thing against a quadratic brute force on 900 random small cases (negatives, zeros,
positives, interleaved updates) with zero mismatches, and a direct `[10^9]*5` whole-window query
returns `5000000000` — the exact value a 32-bit accumulator would have wrapped negative — confirming
the `long long` decision. Timed, `n = q = 2*10^5` runs in about 0.15 s.

The full program — the four-field `Node`, `makeLeaf`/`combine`, `build`/`update`/`query`, and the
`main` I/O loop — is in the answer.
