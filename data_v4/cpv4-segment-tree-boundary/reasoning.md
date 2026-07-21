Two boundaries decide whether this is correct; the segment tree carrying them is standard. A query
asks for the length of the longest **strictly increasing** contiguous block that lies **entirely inside**
`[l, r]` (both ends inclusive), under interleaved point assignments `1 p x`, with `n, q <= 2*10^5`. A
single position is a length-`1` block, so every answer is at least `1`. The naive per-query scan is
`O(nq) = 4*10^{10}` — hopeless — so the structure is forced: a segment tree whose nodes summarise
"runs" and merge in `O(1)`, giving `O((n+q) log n)`. What actually decides correct-vs-wrong is not the
tree but (1) the *seam* where two blocks glue, since a step counts only if it is a **strict** ascent,
so equal neighbours must not join; and (2) the *window edges* `l` and `r`, where a run that continues
past the boundary has to be clipped. Both are inclusive/exclusive hazards, and the plateau-heavy,
small-alphabet hidden tests are aimed straight at them. Values are `|a[i]| <= 10^9`, but I only ever
*compare* them, never sum, so `int` arithmetic would be safe; I keep them in `long long` for a uniform
struct and unambiguous comparisons. Output lengths are at most `n`, well inside `int`. I keep the
brute scan around as an independent oracle even though it is far too slow to submit.

**The node and the merge.** A node covering a contiguous block must support gluing on either side, so
the minimal state is: `best` (longest increasing run fully inside the block), `pre` (longest run
*starting at the block's left end*), `suf` (longest run *ending at the block's right end*), the
boundary values `lval`/`rval`, and `len` (number of positions). A leaf for value `v` is
`Node{lval=v, rval=v, pre=1, suf=1, best=1, len=1}`.

Merging `L` (left block) then `R` (right block) into their concatenation, the only new structure is
the seam between `L`'s right end and `R`'s left end. A run may cross it **iff the seam is a strict
ascent**, `join = (L.rval < R.lval)`. Then:

- `best`: a maximal run inside the concatenation lies wholly in `L`, wholly in `R`, or crosses the
  seam. So `best = max(L.best, R.best)`, and *if* `join`, also consider `L.suf + R.pre`.
- `pre`: the merged prefix starts at `L`'s left end. It equals `L.pre`, *unless* `L` is one unbroken
  increasing run end-to-end (`L.pre == L.len`) **and** the seam joins, in which case it continues into
  `R`: `pre = L.len + R.pre`.
- `suf`: symmetric — `R.suf`, unless `R.suf == R.len && join`, then `R.len + L.suf`.

`lval = L.lval`, `rval = R.rval`, `len = L.len + R.len`. Build merges children bottom-up; a point
update rewrites one leaf and re-merges up its path. On the sample `a = [1,3,2,4,5,6,1,7]` this gives
root `best = 4` for the run `2<4<5<6`; the seam between the `(1,3)` and `(2,4)` subblocks has
`rval=3` vs `lval=2`, does **not** join, so `3` and `2` are never fused into one run.

That leaves two boundary traps hiding in the merge, both of which the plateau-heavy tests would
expose. Written out with a strict `<`:

```
bool join = (L.rval < R.lval);          // seam joins only on a strict ascent
res.pre = L.pre;
if (L.pre == L.len && join) res.pre = L.len + R.pre;
res.suf = R.suf;
if (R.suf == R.len && join) res.suf = R.len + L.suf;
res.best = max(L.best, R.best);
if (join) res.best = max(res.best, L.suf + R.pre);
```

**Trap one: `<` not `<=` at the seam.** The reflex is to write `join = (L.rval <= R.lval)`, but that
treats a flat step as an ascent. On `a = [7,7]`, query `[1,2]`: with `<=`, `join` is true, so
`best = max(1, L.suf + R.pre) = max(1, 2) = 2` — an illegal answer, since `7 < 7` is false and equal
values never extend a strictly-increasing run. With `<`, `join = (7 < 7)` is false, nothing glues, and
the root `best` stays `1`. Every plateau in the array rides on this one character.

**Trap two: the `pre == len` extension guard.** It is tempting to simplify the prefix line to
`if (join) res.pre = L.len + R.pre`, dropping the `L.pre == L.len` test — "if the seam joins, the
prefix reaches into `R`." That is wrong whenever `L`'s prefix was already cut short *before* the seam.
Take `a = [3,1,2]`, whose root splits `[3,1] | [2]`. The left block is `L0 = merge([3],[1])`: seam
`3<1` false, so `L0 = {lval=3, rval=1, pre=1, suf=1, best=1, len=2}`. Merging `L0` with leaf `[2]`,
the seam `1<2` joins; the unguarded line would set `pre = L0.len + 1 = 3`, claiming a length-`3` run
starting at position `1` — but `3>1` breaks the run immediately, so the true prefix is `1`. That bogus
`pre` would then poison every parent merge, over-reporting `best`. The guard `L0.pre == L0.len` is
`1 == 2`, false, so `pre` correctly stays `1`. The guard encodes the real invariant: a prefix crosses
a joined seam **only** when it already spans all of `L`.

**The window edge.** The merge is boundary-correct; the other place a run can leak is the query. I
answer a query by returning a `Node` that summarises **exactly** the intersection of a tree node's
range with `[ql, qr]`, descending only into children that overlap the window and merging their partial
results left-to-right. The left child covers `[lo, mid]`, so it overlaps iff `ql <= mid`; the right
child covers `[mid+1, hi]`, so it overlaps iff `qr > mid`. Because every partial I merge is itself
already clipped to `[ql, qr]`, the seam checks inside `merge` only ever see *in-window* boundary
values — clipping happens by *not descending*, never by post-hoc subtraction. That is what makes the
sample query `[5,8]` on `1 3 2 4 5 6 8 7` return `3` (`5<6<8`) rather than `4`: the longer `4<5<6<8`
starts at position `4`, and no `Node` covering position `4` is ever entered when `ql = 5`, so its `4`
simply never exists in the merged result.

The remaining edge cases fall out of the same two rules. `l == r` reaches a single leaf and returns
`best = 1`. An all-equal or strictly-decreasing array has `join` false at every seam, so every `best`
stays `1` — equal is not an ascent. A strictly-increasing array queried on a proper sub-window returns
the *clipped* length, not `n`, because the run is cut at `l` and `r`. Pure-update inputs print nothing,
since only type-`2` operations emit a line.

**Verification.** I stress-tested the tree against the brute scanner as an independent oracle:
hundreds of random interleaved cases on small arrays with tiny value alphabets, so equal-neighbour
seams and clipped windows occur constantly — zero mismatches, and the documented sample reproduces
exactly (`4 / 4 / 3 / 3`). Both boundary traps were handled in the merge itself, and the stress run
covers what hand-reasoning could miss.

The full self-contained C++ program — build, update, the clipping query, and the strict-`<` guarded
merge above — is in the answer.
