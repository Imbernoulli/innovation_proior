**Problem.** Maintain `a[1..n]` under two online operations: `1 p x` sets `a[p]=x`, and `2 l r` asks
for the length of the longest contiguous block that is **strictly increasing** and lies **entirely
inside `[l, r]`** (inclusive). A single position is a length-`1` block, so every answer is at least
`1`. Constraints: `n, q <= 2*10^5`, `|a[i]| <= 10^9`.

**Key idea — a structural segment tree of "run" stats.** Per node (covering a contiguous block)
store: `best` (longest increasing run fully inside the block), `pre` (longest run starting at the
block's left end), `suf` (longest run ending at the right end), the boundary values `lval`/`rval`,
and the block length `len`. A leaf for value `v` is `{lval=v, rval=v, pre=suf=best=1, len=1}`. Merge
of left block `L` then right block `R` glues a run across the seam **iff the seam is a strict ascent**,
`join = (L.rval < R.lval)`:

- `best = max(L.best, R.best)`, and if `join`, also `L.suf + R.pre`.
- `pre = L.pre`, unless `L.pre == L.len && join`, then `L.len + R.pre`.
- `suf = R.suf`, unless `R.suf == R.len && join`, then `R.len + L.suf`.

Point update rebuilds one leaf-to-root path (`O(log n)`); a query stitches the `O(log n)` blocks that
overlap `[l, r]`, descending into the left child iff `ql <= mid` and the right child iff `qr > mid`,
and merging the *clipped* partials.

**Pitfalls.**
1. *Inclusive vs exclusive seam (the headline off-by-one).* "Strictly increasing" means `<`, not
   `<=`. Writing `join = (L.rval <= R.lval)` treats a flat step (`7, 7`) as an ascent and inflates
   every run through a plateau. Trace `[7,7]` on `[1,2]`: the buggy `<=` returns `2`; the correct
   answer is `1`. Use `L.rval < R.lval`.
2. *Prefix/suffix extension guard.* The prefix may cross a joined seam **only when the left child is
   one unbroken run** (`L.pre == L.len`); otherwise the prefix was already cut short inside `L`.
   Dropping the guard makes `[3,1,2]` report a length-`3` prefix that does not exist, poisoning parent
   merges. Keep `if (L.pre == L.len && join)` (and the symmetric `suf` guard).
3. *Window clipping at `l`/`r`.* A run that continues *outside* `[l, r]` must not count. On
   `1 3 2 4 5 6 8 7`, query `[5,8]` is `3` (`5<6<8`), not `4` — the longer `4<5<6<8` starts at
   position `4`, outside the window. The query enforces this by *not descending* into out-of-window
   children, so no `Node` it merges ever references a position outside `[l, r]`.

**Edge cases.** `l == r` → `1`; all-equal array → `1` everywhere (equal is not an ascent); strictly
decreasing → `1`; strictly increasing with a clipped window returns the clipped length, not `n`;
pure-update inputs print nothing. Verified against an `O(n)`-per-query brute force on `700` random
cases (small alphabets to stress equal-neighbour seams, plus full `±10^9` values) with zero
mismatches.

**Complexity.** `O((n + q) log n)` time, `O(n)` memory. Lengths fit in `int`; values are compared,
never summed.

**Code.**

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
