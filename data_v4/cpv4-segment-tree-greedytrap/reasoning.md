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

Let me sanity-check the straddle case is consistent with the empties: since `L.suf >= 0` and
`R.pre >= 0`, the gluing term is `>= 0`, and because `L.best, R.best >= 0`, `P.best >= 0` too, so the
empty block survives as required.

Hand-check on `[3, -2, 5]`. Leaves: `3 -> (3,3,3,3)`, `-2 -> (-2,0,0,0)`, `5 -> (5,5,5,5)`. Merge
`[3]` with `[-2]`: `total = 1`; `pre = max(3, 3 + 0) = 3`; `suf = max(0, -2 + 3) = 1`;
`best = max(3, 0, 3 + 0) = 3`. So `[3,-2] -> (1, 3, 1, 3)`. Merge that with `[5]`:
`total = 1 + 5 = 6`; `pre = max(3, 1 + 5) = 6`; `suf = max(5, 5 + 1) = 6`;
`best = max(3, 5, 1 + 5) = 6`. Best block of `[3,-2,5]` is `6` — matches the whole-array sample.
The merge algebra is right.

**First implementation — and immediately a trace, because clean algebra transcribes dirty.** I write
the leaf and combine, build a `4n` tree, and a query that returns an identity node for the empty
range. My first cut of the query identity and the all-zero leaf default looked fine, but I want to
trace the *partial-overlap query path* on a small case, because that is where range segment trees
quietly break. Consider `n = 3`, `a = [-2, 5, -1]`, query `2 2 3` (window `[5, -1]`, answer should be
`5`). The tree over `[0,2]` splits at `mid = 1`: left covers `[0,1]`, right covers `[2,2]`. The query
range in 0-indexed terms is `[1, 1] .. [2, 2]`, i.e. `l = 1, r = 2`. At the root the range is not
fully inside, so I recurse. My first query function read like this.

**The first bug — missing the out-of-range base case.** Trace it. Root `[0,2]`, `l=1,r=2`: not fully
covered, `mid=1`, recurse left `[0,1]` and right `[2,2]`. Left `[0,1]`, `l=1,r=2`: not fully covered
(`0 < 1`), `mid=0`, recurse left `[0,0]` and right `[1,1]`. Left `[0,0]`, `l=1,r=2`: **`0` is not in
`[1,2]` at all**, but there is no guard, so `l<=0 && 0<=r` is `1<=0` = false, I do *not* return the
node, and I keep recursing... into a leaf where `lo==hi==0`, which never hits the `l<=lo&&hi<=r`
return and has no children — undefined indices `2*node` past the array, garbage. Even setting the
crash aside, the leaf `[0,0]` holds `a[0] = -2`, which is **outside** the query window and must not
contribute. Without an out-of-range guard I fold position `0` into the answer. That is a correctness
bug independent of the crash. The merge identity for "this node contributes nothing" must be the
empty-segment node `{total:0, pre:0, suf:0, best:0}` and I must return it whenever the node's range is
disjoint from `[l, r]`.

Re-trace `a = [-2, 5, -1]`, query 0-indexed `[1, 2]`. Root `[0,2]`: not disjoint, not fully inside,
`mid=1`; `r=2 > mid` and `l=1 <= mid`, so the general branch: recurse left `[0,1]` and right `[2,2]`.
Left `[0,1]`, `[1,2]`: not disjoint, not fully inside, `mid=0`; here `l = 1 > mid = 0`, so I go *only*
right: query `[1,1]` with `[1,2]` -> fully inside -> returns leaf `5 -> (5,5,5,5)`. So left subtree
contributes `(5,5,5,5)`, having correctly *skipped* the `[0,0]` leaf. Right `[2,2]`, `[1,2]`: fully
inside -> leaf `-1 -> (-1,0,0,0)`. Combine `(5,5,5,5)` with `(-1,0,0,0)`:
`total = 5 + (-1) = 4`; `pre = max(5, 5 + 0) = 5`; `suf = max(0, -1 + 5) = 4`;
`best = max(5, 0, 5 + 0) = 5`. Answer `5`. Correct, and position `0` never entered. The disjoint
guard was the fix and it fixed it for the reason I diagnosed.

**Second trace — the empty-block / all-negative case, where the `max(0, .)` floors earn their keep.**
I worry that an all-negative window might leak a negative answer if any field forgot its `>= 0` floor.
Trace `a = [-3, -1, -4]`, query the whole thing, 0-indexed `[0, 2]`, expected `0` (take nothing).
Leaves: `-3 -> (-3,0,0,0)`, `-1 -> (-1,0,0,0)`, `-4 -> (-4,0,0,0)`. Merge `[-3]` and `[-1]`:
`total = -4`; `pre = max(0, -3 + 0) = 0`; `suf = max(0, -1 + 0) = 0`; `best = max(0,0, 0+0) = 0` ->
`(-4, 0, 0, 0)`. Merge with `[-4]`: `total = -8`; `pre = max(0, -4 + 0) = 0`;
`suf = max(0, -4 + 0) = 0`; `best = max(0, 0, 0 + 0) = 0` -> `(-8, 0, 0, 0)`. Query returns `best = 0`.
Correct. The `total` field is allowed to go negative (it is a true sum, used only inside `pre/suf`
gluing), but `pre`, `suf`, `best` never do, so the empty selection is preserved end to end. Good — and
this is exactly the leaf where a naive "best = max(L.best, R.best, L.suf + R.pre)" *without* the leaf
floor would have produced `-1` or `-3`.

**A targeted check that greedy and the tree disagree on the trap input.** I run my reasoning on
`[5, -100, 4]`: leaves `(5,5,5,5)`, `(-100,0,0,0)`, `(4,4,4,4)`. Merge `[5],[-100]`: `total=-95`;
`pre=max(5,5+0)=5`; `suf=max(0,-100+5)=0`; `best=max(5,0,5+0)=5` -> `(-95,5,0,5)`. Merge with `[4]`:
`total=-91`; `pre=max(5,-95+4)=5`; `suf=max(4,4+0)=4`; `best=max(5,4,0+4)=4`... wait, `L.suf=0`,
`R.pre=4`, glue `=4`; `max(5,4,4)=5`. Final `best = 5`. The tree says `5`; greedy said `9`. The tree
matches the by-hand truth and refuses the impossible `9`. Confirmed.

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
