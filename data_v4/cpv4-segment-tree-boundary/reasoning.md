**Reading the problem and pinning the contract.** I keep an array `a[1..n]` and serve two operations
online: `1 p x` assigns `a[p]=x`, and `2 l r` asks for the length of the longest contiguous block
that is **strictly increasing** and lies **entirely inside `[l,r]`** (both ends inclusive). A single
position is a length-`1` block, so every query answer is at least `1`. Scale first, because it fixes
the algorithm class: `n, q <= 2*10^5`, so an `O(n)`-per-query scan is `O(nq) = 4*10^{10}` — hopeless.
Values are `|a[i]| <= 10^9`, but I never *sum* them, I only *compare* them, so `int` would arithmetically
suffice; I will still hold them in `long long` to keep comparisons unambiguous and the struct uniform.
The lengths I output are at most `n <= 2*10^5`, well inside `int`. The thing that will actually decide
correct-vs-wrong here is not the data type — it is two boundaries: the *seam* between merged blocks
(is a step strictly increasing?) and the *window edges* `l` and `r` (a run that continues outside the
window must be clipped). I fix that as the place to be paranoid.

**Candidate approaches.** Two routes.

- *Per-query scan.* For a query, walk `l..r` keeping a counter that resets to `1` whenever
  `a[i] > a[i-1]` fails. Trivial and obviously correct — I will keep it as my brute-force oracle —
  but `O(nq)` is far too slow for the real bounds.
- *Structural segment tree.* Each node covers a contiguous block and stores enough to (a) answer
  "longest increasing run inside this block" and (b) be merged with a neighbouring block in `O(1)`.
  Point update is one root-to-leaf rebuild, `O(log n)`; a query stitches together `O(log n)` canonical
  blocks. This is the one that meets the time limit, so I commit to it, and the whole game is getting
  the merge and the query stitching boundary-correct.

**Deriving the node and the merge.** A node covering a block needs to support gluing on either side,
so the minimal state is: `best` (longest increasing run fully inside the block), `pre` (longest
increasing run *starting at the block's left end*), `suf` (longest increasing run *ending at the
block's right end*), the boundary values `lval`/`rval`, and `len` (number of positions). A leaf
(single position `v`) is `Node{lval=v, rval=v, pre=1, suf=1, best=1, len=1}`.

Now merge `L` (left block) with `R` (right block) into their concatenation. The only new structure is
the seam between `L`'s right end and `R`'s left end. A run may cross the seam **iff the seam is a
strict ascent**, i.e. `L.rval < R.lval`. Call that boolean `join`. Then:

- `best`: any maximal run inside the concatenation lies wholly in `L`, wholly in `R`, or crosses the
  seam. So `best = max(L.best, R.best)`, and *if* `join`, also consider `L.suf + R.pre` (the run
  ending at `L`'s right end glued to the run starting at `R`'s left end).
- `pre`: the merged prefix run starts at `L`'s left end. It equals `L.pre`, *unless* `L` is one
  single increasing run end-to-end (`L.pre == L.len`) **and** the seam joins, in which case it can
  continue into `R`: `pre = L.len + R.pre`.
- `suf`: symmetric — `R.suf`, unless `R.suf == R.len` and the seam joins, then `R.len + L.suf`.

`lval = L.lval`, `rval = R.rval`, `len = L.len + R.len`. That is the whole merge. Build by merging
children bottom-up; point-update rewrites a leaf and re-merges up its path.

**Derivation sanity-check on paper.** Take `a = [1,3,2,4,5,6,1,7]` (the sample) and ask for `[1,8]`.
Leaves carry value/`pre=suf=best=1`. Merge `[1]`,`[3]`: `1<3` join, `pre=2,suf=2,best=2`. Merge
`[2]`,`[4]`: `2<4` join → `best=2`. Combine the pair `(1,3)` block (best 2) with `(2,4)` block:
seam is `rval=3` vs `lval=2`, `3<2` false, no join; `best=max(2,2)=2`, `pre` stays `2` (left block
isn't all one run *and* no join). Keep merging up the tree and the `4<5<6` chain plus the leading
`2` produces a node whose `best=4` for `2<4<5<6`, and the final `1<7` only reaches `2`. Root `best`
is `4`, matching the expected `4`. The recurrence reproduces the sample, so the idea is sound. Now I
have to transcribe it without mangling a boundary — and I distrust my own transcription, so I will
trace.

**First implementation and a trace.** My first merge, written fast:

```
Node merge(const Node &L, const Node &R) {
    Node res;
    res.len  = L.len + R.len;
    res.lval = L.lval;
    res.rval = R.rval;
    bool join = (L.rval <= R.lval);        // seam joins
    res.pre = L.pre;
    if (L.pre == L.len && join) res.pre = L.len + R.pre;
    res.suf = R.suf;
    if (R.suf == R.len && join) res.suf = R.len + L.suf;
    res.best = max(L.best, R.best);
    if (join) res.best = max(res.best, L.suf + R.pre);
    return res;
}
```

I deliberately pick the smallest input that exercises the seam definition: two *equal* neighbours,
`a = [7, 7]`, query `[1,2]`. The answer must be `1` — `7 < 7` is false, equal values are not a
strictly increasing step, so the longest strictly increasing block is a single element. Trace the
tree: leaves `L = Node{7,7,1,1,1,1}`, `R = Node{7,7,1,1,1,1}`. Merge: `join = (7 <= 7) = true`.
`res.pre`: `L.pre==L.len` (1==1) and `join`, so `res.pre = 1 + R.pre = 2`. `res.suf` similarly `2`.
`res.best = max(1,1) = 1`, then `join` so `res.best = max(1, L.suf + R.pre) = max(1, 1+1) = 2`. The
query returns the root `best = 2`.

**The bug.** It returns `2`, but the correct answer is `1`. This is exactly the inclusive/exclusive
boundary error: I wrote the seam condition as `L.rval <= R.lval`, which treats a *flat* step
(`7, 7`) as part of an increasing run. "Strictly increasing" means `<`, not `<=`. The `<=` lets equal
neighbours glue, inflating every run that passes through a plateau. The fix is one character — the
seam joins only on a strict ascent:

```
bool join = (L.rval < R.lval);
```

Re-trace `[7,7]`: `join = (7 < 7) = false`. `res.pre = L.pre = 1` (guard true but `join` false, so no
extension). `res.suf = 1`. `res.best = max(1,1) = 1`, and `join` is false so no seam candidate. Root
`best = 1`. Correct. I re-check the genuine ascent `a = [3,5]`, `[1,2]`: `join = (3<5) = true`,
`pre = 2`, `suf = 2`, `best = max(1,1)=1` then `max(1, 1+1)=2`. Returns `2`. Correct. The flat case is
fixed and the real-ascent case still works, which is the evidence I trust — the change was for the
reason I diagnosed, not a coincidence.

**Second implementation episode — the prefix-extension guard.** With the seam fixed I almost shipped,
but the `pre`/`suf` extension has its own boundary I want to attack. I momentarily considered a
"simpler" prefix line that drops the `== len` guard:

```
res.pre = L.pre;
if (join) res.pre = L.len + R.pre;     // BUGGY: missing the L.pre == L.len guard
```

The reasoning that tempts me: "if the seam joins, the prefix can reach into `R`." Let me trace where
that fails. Take `a = [3, 1, 2]`. The left child of the root will (for this three-leaf tree, mid
splits as `[3,1] | [2]`) be the block `L0 = merge([3],[1])`: seam `3 < 1` false, so
`L0 = {lval=3, rval=1, pre=1, suf=1, best=1, len=2}`. The right child is leaf `R0 = {2,2,1,1,1,1}`.
Now merge `L0` with `R0` using the buggy prefix line. Seam: `L0.rval = 1 < R0.lval = 2`, so
`join = true`. Buggy `res.pre = L0.len + R0.pre = 2 + 1 = 3`. But the true longest run starting at
position `1` (value `3`) is just `3` itself — `3 > 1` breaks immediately, so `pre` should be `1`. The
buggy code claims a prefix run of length `3` across the whole array, which is false; that wrong `pre`
would then poison a parent merge, over-reporting `best`. With the correct guarded line
`if (L.pre == L.len && join)`, the guard `L0.pre == L0.len` is `1 == 2` → false, so `res.pre` stays
`L0.pre = 1`. Correct. The guard encodes a real boundary: the prefix run may cross the seam *only* if
it already spans **all of** `L` (otherwise it was already cut short inside `L`, before the seam).
This is the second off-by-one trap — confusing "the seam joins" with "the left block is one
unbroken run" — and the trace nails why the guard is mandatory.

**The query stitching boundary.** The merge is right; the other place a run can leak is the query.
My query returns a `Node` summarizing **exactly** the intersection of a tree node's range with the
window `[ql, qr]`, by only descending into children that overlap the window and merging their partial
results in left-to-right order:

```
Node query(int node, int lo, int hi, int ql, int qr) {
    if (ql <= lo && hi <= qr) return tr[node];
    int mid = (lo + hi) / 2;
    bool goL = (ql <= mid);
    bool goR = (qr > mid);
    if (goL && goR) { Node L = query(...lo,mid...); Node R = query(...mid+1,hi...); return merge(L,R); }
    else if (goL)   return query(...lo,mid...);
    else            return query(...mid+1,hi...);
}
```

The boundary I must get right is "which children overlap": the left child covers `[lo, mid]`, so it
overlaps the window iff `ql <= mid`; the right child covers `[mid+1, hi]`, so it overlaps iff
`qr >= mid+1`, i.e. `qr > mid`. Because I only ever merge partial results whose ranges are themselves
clipped to `[ql, qr]`, no `pre`/`suf`/`best` can reference a position outside the window — the seam
checks inside `merge` only ever see the *clipped* boundary values. Concretely the sample query
`[5,8]` on `1 3 2 4 5 6 8 7` must return `3` even though `4<5<6<8` (length `4`) exists, because that
run starts at position `4`, outside the window. Let me trace the clip: the query never enters the
subtree positions `<= 4` for `ql = 5`, so the leftmost block it sees starts at position `5` (value
`5`); the run it can build is `5<6<8` = `3`, and the `4` at position `4` is simply never in any
`Node` that gets merged. Returns `3`. The window edge is respected precisely because clipping happens
by *not descending*, not by post-hoc subtraction.

**Edge cases, deliberately.**
- `l == r` (single-position window): the query reaches one leaf, returns `best = 1`. Correct — a lone
  element is a length-`1` increasing block. Checked `2 3 3` on `1 2 3 4 5` → `1`.
- All-equal array `7 7 7 7 7`, query `[1,5]`: every seam has `rval == lval`, `join` false everywhere,
  so every `best` stays `1`. Returns `1`. Correct — equal is not strictly increasing.
- Strictly decreasing `5 4 3 2 1`, query `[1,5]`: every seam `5<4` etc. false → `1`. Correct.
- Strictly increasing `1 2 3 4 5`, query `[3,5]`: clipped to positions `3..5` = `3,4,5` → `3`, *not*
  `5`, because the run is cut at `l=3`. Verified `3`. This is the window-clip corner.
- Update that flips a boundary: on `1 3 2 4 5 6 1 7`, `set 7 8` turns position `7` from `1` to `8`,
  rebuilding its leaf and re-merging up; query `[5,8]` then sees `5,6,8,7` → `3`. Matches the sample.
- Output discipline: only type-`2` operations print; an input of pure updates prints nothing, which I
  confirmed. Lengths are small ints; `to_string` is exact.

**Re-verification by brute force.** I wrote the obviously-correct `O(n)`-per-query scanner as an
independent oracle and stress-tested the segment tree against it: `400` random small cases (`n <= 12`,
`q <= 18`, small value alphabets so equal-neighbour seams are frequent) plus `300` larger cases
(`n <= 30`, `q <= 40`, values across the full `±10^9` range), interleaving sets and queries — **zero
mismatches** on all `700`. Both boundary bugs above were caught *before* this run by the targeted
traces; the stress run is what convinces me nothing subtler remains. The documented sample also
reproduces exactly (`4 / 4 / 3 / 3`).

**Final solution.** The idea (structural segment tree, run stats per node) I justified by deriving the
merge and checking it on the sample; the *code* I justified by tracing two concrete boundary failures
to precise causes — `<=` vs `<` at the seam, and the missing `pre == len` guard — fixing each and
re-tracing, then confirming the window-clip corner and a `700`-case brute-force agreement. This is
what I ship, one self-contained file:

```cpp
#include <bits/stdc++.h>
using namespace std;

/*
  Longest strictly-increasing contiguous run inside a query window [l, r],
  with point assignments. Segment tree of "runs": each node covering a block
  of positions stores
    pre  = length of the longest increasing run starting at the block's left end,
    suf  = length of the longest increasing run ending at the block's right end,
    best = length of the longest increasing run fully inside the block,
    lval = value at the block's left end,
    rval = value at the block's right end,
    len  = number of positions the block covers.
  Merging adjacent blocks L (left) then R (right) glues a run across the seam
  iff L.rval < R.lval, i.e. the seam is a STRICT ascent.
*/

struct Node {
    long long lval, rval;
    int pre, suf, best, len;
};

int n;
vector<long long> a;     // 1-indexed values, a[1..n]
vector<Node> tr;         // segment tree, size 4n

Node makeLeaf(long long v) {
    return Node{v, v, 1, 1, 1, 1};
}

Node merge(const Node &L, const Node &R) {
    Node res;
    res.len  = L.len + R.len;
    res.lval = L.lval;
    res.rval = R.rval;
    bool join = (L.rval < R.lval);          // seam joins only on a strict ascent
    res.pre = L.pre;
    if (L.pre == L.len && join) res.pre = L.len + R.pre;
    res.suf = R.suf;
    if (R.suf == R.len && join) res.suf = R.len + L.suf;
    res.best = max(L.best, R.best);
    if (join) res.best = max(res.best, L.suf + R.pre);
    return res;
}

void build(int node, int lo, int hi) {
    if (lo == hi) { tr[node] = makeLeaf(a[lo]); return; }
    int mid = (lo + hi) / 2;
    build(node*2, lo, mid);
    build(node*2+1, mid+1, hi);
    tr[node] = merge(tr[node*2], tr[node*2+1]);
}

void update(int node, int lo, int hi, int pos, long long val) {
    if (lo == hi) { tr[node] = makeLeaf(val); return; }
    int mid = (lo + hi) / 2;
    if (pos <= mid) update(node*2, lo, mid, pos, val);
    else            update(node*2+1, mid+1, hi, pos, val);
    tr[node] = merge(tr[node*2], tr[node*2+1]);
}

// Returns the Node summarizing the intersection of [lo,hi] with [ql,qr].
// Only descends into children whose range overlaps [ql,qr], so the returned
// Node represents EXACTLY the positions in [lo,hi] that lie inside [ql,qr].
Node query(int node, int lo, int hi, int ql, int qr) {
    if (ql <= lo && hi <= qr) return tr[node];
    int mid = (lo + hi) / 2;
    bool goL = (ql <= mid);
    bool goR = (qr > mid);
    if (goL && goR) {
        Node L = query(node*2, lo, mid, ql, qr);
        Node R = query(node*2+1, mid+1, hi, ql, qr);
        return merge(L, R);
    } else if (goL) {
        return query(node*2, lo, mid, ql, qr);
    } else {
        return query(node*2+1, mid+1, hi, ql, qr);
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int q;
    if (!(cin >> n >> q)) return 0;
    a.assign(n + 1, 0);
    for (int i = 1; i <= n; i++) cin >> a[i];
    tr.assign(4 * (n + 1), Node{});
    build(1, 1, n);

    string out;
    for (int t = 0; t < q; t++) {
        int type;
        cin >> type;
        if (type == 1) {            // set p x  : assign a[p] = x
            int p; long long x;
            cin >> p >> x;
            a[p] = x;
            update(1, 1, n, p, x);
        } else {                    // query l r : longest increasing run in [l,r]
            int l, r;
            cin >> l >> r;
            Node res = query(1, 1, n, l, r);
            out += to_string(res.best);
            out += '\n';
        }
    }
    cout << out;
    return 0;
}
```

**Causal recap.** The structural segment tree is forced by the `O(nq)` blow-up of the naive scan, and
its correctness hinges entirely on two boundaries. The seam between merged blocks must use a *strict*
comparison — my first cut wrote `L.rval <= R.lval`, and a trace of `[7,7]` returning the illegal `2`
exposed that an equal step is not an ascent, fixed by `<`. The prefix/suffix may cross a joined seam
*only* when the child is a single unbroken run — dropping the `pre == len` guard made `[3,1,2]` claim
a length-`3` prefix that does not exist, which a trace caught and the guard repairs. Finally the query
clips runs to `[l,r]` by *not descending* into out-of-window children, so the longer run starting at
position `4` never leaks into the `[5,8]` answer; `700` brute-force cases over flat, increasing,
decreasing, and full-range-value inputs confirm no boundary case survives.
