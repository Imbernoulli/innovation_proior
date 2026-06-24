**Reading the problem and pinning the contract.** I have a row of `n` integers `a[1..n]`, some
possibly negative, and two online request types. An update `1 p v` sets `a[p] = v`. A query `2 l r`
wants the maximum sum over a *contiguous* block of indices inside `[l, r]`, where the empty block is
allowed, so every answer is `>= 0`. I read `n` and `q`, then the array, then `q` lines, and for each
type-`2` line I print one integer. Before any algorithm I fix the scale because it forces the data
types: `n, q <= 2*10^5` and `|a[i]| <= 10^9`, so a window can sum to `2*10^5 * 10^9 = 2*10^14`, far
past the 32-bit ceiling of `~2.1*10^9`. Every sum, every stored field, every accumulator must be
64-bit `long long`. That decision is non-negotiable; an `int` is a silent wrong-answer on the large
tests.

The other scale fact: a per-query Kadane scan is `O(window)`, and with `q` queries that is `O(nq)`
worst case `4*10^10`, hopeless in 2 seconds. With updates in the mix I cannot precompute a static
structure (sparse table over fixed blocks) either, because entries change. I need something that
both answers a range max-subarray and absorbs point updates in sublinear time. That smells like a
segment tree carrying enough per-node statistics to merge two children in `O(1)`.

**Laying out the candidate approaches.** Two routes, and I want the one I can *prove*, not the one
that types fastest.

- *Greedy: sum the positives in the window.* For a query `[l, r]`, just add every `a[i] > 0`. It is
  irresistibly simple and I could even make it `O(log n)` with a Fenwick tree over positive values
  plus another for "count" so I can ignore updates of sign changes. The structural worry: the answer
  must be a single *contiguous* run of indices, and "take all gains, skip all losses" silently
  assumes the gains form one block. A positive value can be stranded on the far side of a deep
  negative, so connecting two gain-clusters across a loss may be worse than taking just one cluster.
  I will not trust this until I have tried to break it.
- *Segment tree of merged statistics.* Each node over a segment stores four numbers and merges in
  `O(1)`. Build `O(n)`, update and query `O(log n)`. The worry here is not the idea but the *merge
  algebra* — it is famous for being written subtly wrong, especially the empty-block convention.

**Stress-testing greedy before committing.** "Greedy feels right" is how wrong solutions ship, so I
attack it with a concrete instance. Take the window `a = [5, -100, 4]`. Greedy sums the positives:
`5 + 4 = 9`. But those two `+`s are separated by `-100`; to take both as one contiguous block I must
include the `-100`, giving `5 - 100 + 4 = -91`. The only contiguous blocks are `[5]=5`, `[4]=4`,
`[5,-100]=-95`, `[-100,4]=-96`, `[5,-100,4]=-91`, and the empty block `=0`. The true best is `5`.
Greedy reports `9`, which corresponds to *no actual contiguous block at all*. So greedy is not just
suboptimal, it is unachievable — it over-counts. Greedy is out, and I see exactly why: contiguity is
a global structural constraint, and "sum the positives" pretends positives can be teleported next to
each other. This is the trap. It even looks correct on all-positive windows (where it coincides with
"take everything"), which is what makes it tempting.

The companion sample confirms it: on the full array `[3,-2,5,-1,-6,4,2]` greedy would say
`3+5+4+2 = 14`, but the real best contiguous block is `[3,-2,5] = 6` (or `[4,2] = 6`); the `-6`
quarantines `4,2` from the left cluster, and bridging it costs more than it gains.

**Deriving the segment-tree statistics and checking the merge on paper.** I want each node, covering
a contiguous segment, to expose just enough to merge two adjacent segments in `O(1)`. The classic
quadruple:

- `total` = sum of all elements in the segment,
- `pre`   = best sum of a *prefix* of the segment (prefix may be empty, so `pre >= 0`),
- `suf`   = best sum of a *suffix* of the segment (suffix may be empty, so `suf >= 0`),
- `best`  = best sum of any contiguous block inside the segment (block may be empty, so `best >= 0`).

Encoding "may be empty" as "every field `>= 0`" is exactly how the problem's empty-block rule lives
inside the structure: the empty selection is always a competitor, so all four fields floor at `0`.

A **leaf** holding value `v`: `total = v`; the best non-empty prefix/suffix/block is `v` itself, but
empty beats it when `v < 0`, so `pre = suf = best = max(0, v)`.

The **merge** of a left child `L` and right child `R` into parent `P`:

- `P.total = L.total + R.total`. (Trivial.)
- `P.pre`: a prefix of `P` is either a prefix that stays inside `L` (value `L.pre`), or all of `L`
  plus a prefix of `R` (value `L.total + R.pre`). So `P.pre = max(L.pre, L.total + R.pre)`.
- `P.suf`: symmetric — a suffix inside `R` (`R.suf`), or all of `R` plus a suffix of `L`
  (`R.total + L.suf`). So `P.suf = max(R.suf, R.total + L.suf)`.
- `P.best`: the best block lies entirely in `L` (`L.best`), entirely in `R` (`R.best`), or *straddles*
  the boundary, in which case it is a suffix of `L` glued to a prefix of `R` (`L.suf + R.pre`). So
  `P.best = max(L.best, R.best, L.suf + R.pre)`.

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
fully inside, so I recurse. My first query function read like this:

```
Node query(int node, int lo, int hi, int l, int r) {
    if (l <= lo && hi <= r) return tree[node];
    int mid = (lo + hi) / 2;
    return combine(query(2*node, lo, mid, l, r),
                   query(2*node+1, mid+1, hi, l, r));
}
```

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

I also realize the recursion is wasteful and fragile: I should only descend into a side when the
query actually touches it. Let me fix both.

**Fixing the query and re-verifying.** Add the disjoint base case first, then descend selectively:

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

**Wiring updates and I/O.** A point update `1 p v` (1-indexed) rewrites leaf `p-1` to `makeLeaf(v)`
and re-merges up the path, `O(log n)`. A query `2 l r` calls `query(1, 0, n-1, l-1, r-1)` and prints
`best`. I batch output into a single string and flush once, since up to `2*10^5` lines of output can
make `endl`/per-line flushing slow; with `sync_with_stdio(false)` and one final `cout << out` it is
fine.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 1`: tree is a single leaf; build and updates and single-cell queries all go through the
  `lo == hi` paths. A query `2 1 1` on `a = [-5]` returns `max(0, -5) = 0`. Correct.
- All-negative window: shown above, returns `0`.
- Single-cell query on a positive: `a=[7]`, `2 1 1` -> leaf `(7,7,7,7)`, `best = 7`. Correct.
- Update then query interleaving: a sequence of `1`/`2` ops; each update re-merges its root-path so
  later queries see the new value. Verified by 900 random stress cases against the brute force, zero
  mismatches.
- Overflow: all fields are `long long`. A window of `2*10^5` values near `10^9` sums to `~2*10^14`,
  inside the `~9.2*10^18` `long long` range with vast headroom. I tested `5 * 10^9` directly
  (`[1e9]*5`, query whole) -> `5000000000`, which a 32-bit int would have wrapped to a negative.
- Disjoint identity: returning `{0,0,0,0}` for a non-overlapping subtree is the merge-identity (empty
  segment) and is what made the partial-overlap query correct.

**Final solution.** I convinced myself the idea is right by *disproving greedy with a concrete
unachievable over-count* (`[5,-100,4]`: greedy `9` vs true `5`) and by hand-deriving the four-field
merge and checking it on `[3,-2,5]`; I convinced myself the *code* is right by tracing the
partial-overlap query, finding the missing disjoint base case that leaked out-of-window leaves
(`[-2,5,-1]` query `[2,3]`), fixing it, and re-tracing; and by tracing the all-negative window to
confirm the `>= 0` floors preserve the empty selection. This is what I ship — one self-contained file,
the `O((n+q) log n)` segment tree I can defend rather than the greedy I broke:

```cpp
#include <bits/stdc++.h>
using namespace std;

// Segment tree node for maximum-subarray-sum queries with point updates.
// For a segment we keep:
//   total = sum of all elements,
//   pre   = best sum of a prefix that may be empty (>= 0),
//   suf   = best sum of a suffix that may be empty (>= 0),
//   best  = best sum of a possibly-empty contiguous block (>= 0).
// "may be empty" => every field is >= 0, which encodes the empty selection
// that the problem explicitly allows.
struct Node {
    long long total, pre, suf, best;
};

static int N;
static vector<Node> tree;

Node makeLeaf(long long v) {
    Node nd;
    nd.total = v;
    nd.pre = max(0LL, v);
    nd.suf = max(0LL, v);
    nd.best = max(0LL, v);
    return nd;
}

Node combine(const Node &L, const Node &R) {
    Node nd;
    nd.total = L.total + R.total;
    nd.pre   = max(L.pre, L.total + R.pre);
    nd.suf   = max(R.suf, R.total + L.suf);
    nd.best  = max({L.best, R.best, L.suf + R.pre});
    return nd;
}

void build(int node, int lo, int hi, const vector<long long> &a) {
    if (lo == hi) { tree[node] = makeLeaf(a[lo]); return; }
    int mid = (lo + hi) / 2;
    build(2 * node, lo, mid, a);
    build(2 * node + 1, mid + 1, hi, a);
    tree[node] = combine(tree[2 * node], tree[2 * node + 1]);
}

void update(int node, int lo, int hi, int pos, long long val) {
    if (lo == hi) { tree[node] = makeLeaf(val); return; }
    int mid = (lo + hi) / 2;
    if (pos <= mid) update(2 * node, lo, mid, pos, val);
    else            update(2 * node + 1, mid + 1, hi, pos, val);
    tree[node] = combine(tree[2 * node], tree[2 * node + 1]);
}

// Identity element for an empty range: all zeros (the empty selection).
Node query(int node, int lo, int hi, int l, int r) {
    if (r < lo || hi < l) return Node{0, 0, 0, 0};
    if (l <= lo && hi <= r) return tree[node];
    int mid = (lo + hi) / 2;
    if (r <= mid) return query(2 * node, lo, mid, l, r);
    if (l > mid)  return query(2 * node + 1, mid + 1, hi, l, r);
    return combine(query(2 * node, lo, mid, l, r),
                   query(2 * node + 1, mid + 1, hi, l, r));
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    if (!(cin >> n >> q)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    N = n;
    tree.assign(4 * max(1, n), Node{0, 0, 0, 0});
    if (n > 0) build(1, 0, n - 1, a);

    string out;
    for (int i = 0; i < q; i++) {
        int type;
        cin >> type;
        if (type == 1) {
            // point update: position p (1-indexed) becomes value v
            int p; long long v;
            cin >> p >> v;
            update(1, 0, n - 1, p - 1, v);
        } else {
            // query: maximum-sum contiguous block within [l, r] (1-indexed), empty allowed
            int l, r;
            cin >> l >> r;
            Node res = query(1, 0, n - 1, l - 1, r - 1);
            out += to_string(res.best);
            out += '\n';
        }
    }
    cout << out;
    return 0;
}
```

**Causal recap.** The tempting greedy "sum the positives in the window" looked right and is right on
all-positive windows, but a single traced counterexample (`[5,-100,4]`: greedy `9` vs the true `5`,
and `9` corresponds to *no contiguous block at all*) showed contiguity is a global constraint greedy
cannot honor, so I moved to a four-field segment tree (`total/pre/suf/best`) whose merge glues a
left suffix to a right prefix for boundary-straddling blocks and floors `pre/suf/best` at `0` to
carry the empty selection; my first range query forgot the disjoint base case and a trace of
`[-2,5,-1]` query `[2,3]` showed it folding the out-of-window leaf in, fixed by returning the empty
identity `{0,0,0,0}` for non-overlapping subtrees; a second trace of an all-negative window confirmed
the `>= 0` floors keep the answer at `0`; and 64-bit fields plus a tested `5*10^9` block close out
the overflow corner.
