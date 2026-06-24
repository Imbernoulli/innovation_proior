**Problem.** Maintain `a[0..n-1]` (values may be negative or zero) under two interleaved operations: `1 i x` sets `a[i]=x`, and `2 l r` reports the maximum sum of a **non-empty** contiguous subarray inside `a[l..r]`. The empty subarray is *not* a legal answer, so an all-negative range returns its least-bad single element (negative), an all-zero range returns `0`, and a positive-containing range returns a positive. Read `n, q`, the array, then `q` operations from stdin; print one line per type-`2` query.

**Key idea — segment tree of subarray summaries.** Each node over a range stores four `long long`s:
- `tot` = sum of the whole range;
- `pre` = best non-empty prefix sum (subarray starting at the left end);
- `suf` = best non-empty suffix sum (subarray ending at the right end);
- `best` = best non-empty subarray sum anywhere in the range.

Two adjacent summaries `L`, `R` fuse in `O(1)`:
- `tot  = L.tot + R.tot`
- `pre  = max(L.pre, L.tot + R.pre)`
- `suf  = max(R.suf, R.tot + L.suf)`
- `best = max(L.best, R.best, L.suf + R.pre)`  (left-only, right-only, or straddling the boundary)

A leaf for value `v` is `Node{v, v, v, v}`. Build is `O(n)`, update and query are `O(log n)`.

**Pitfalls (the crux, all on the negatives/zeros corner).**
1. *Identity element.* The out-of-range / empty node must be `Node{pre=NEG, suf=NEG, tot=0, best=NEG}`, **not** `Node{0,0,0,0}`. A zero identity injects a phantom empty subarray of value `0` that beats every real candidate on an all-negative range — e.g. `[-5,-2,-8]` query `2 0 2` wrongly returns `0` instead of `-2`. The empty piece sums nothing (`tot=0`) but can never host a non-empty subarray, so `pre/suf/best` must be a very negative sentinel that never wins a max.
2. *Leaf sign.* The leaf must be the **unclamped** `{v,v,v,v}`. "Defensively" clamping with `max(v,0)` re-injects the empty subarray: `[-7]` query `2 0 0` would return `0` instead of `-7`. Non-empty is mandatory, so the lone element keeps its sign.
3. *Sentinel safety.* Use `NEG = LLONG_MIN/4`, not `LLONG_MIN`. The merge adds to `pre/suf` (e.g. `L.tot + R.pre`, `L.suf + R.pre`); `LLONG_MIN + (negative)` underflows and can wrap to a huge positive that corrupts `best`. `LLONG_MIN/4` leaves headroom so every `NEG + real` stays safely negative.
4. *Overflow.* With `n` up to `2*10^5` and `|a[i]|` up to `10^9`, a range total reaches `~2*10^14`; all four fields must be `long long`. An `int` is a silent wrong-answer.

**Edge cases.** All-negative range -> largest single (negative) element; all-zero range -> `0` (from genuine `{0,0,0,0}` leaves, distinct from the empty identity); single negative element -> that element; `l == r` -> the single element; updates that flip a range all-negative and back -> recombination rebuilds every ancestor, no stale state; empty input guarded by the `cin` check.

**Complexity.** Build `O(n)`; each update and query `O(log n)`; total `O((n + q) log n)` time, `O(n)` memory.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

struct Node {
    long long pre, suf, tot, best;
};

const long long NEG = LLONG_MIN / 4; // sentinel for "no element": best/pre/suf impossible

// Identity for merge: an empty segment. tot = 0 (sums nothing); pre/suf/best = NEG
// because a non-empty subarray cannot be formed from nothing, so it must never win a max.
Node identity() { return Node{NEG, NEG, 0, NEG}; }

Node leaf(long long v) { return Node{v, v, v, v}; }

Node merge(const Node &L, const Node &R) {
    Node res;
    res.tot  = L.tot + R.tot;
    res.pre  = max(L.pre, L.tot + R.pre);
    res.suf  = max(R.suf, R.tot + L.suf);
    res.best = max(max(L.best, R.best), L.suf + R.pre);
    return res;
}

int n, q;
vector<Node> tree;
vector<long long> a;

void build(int node, int lo, int hi) {
    if (lo == hi) { tree[node] = leaf(a[lo]); return; }
    int mid = (lo + hi) / 2;
    build(node * 2, lo, mid);
    build(node * 2 + 1, mid + 1, hi);
    tree[node] = merge(tree[node * 2], tree[node * 2 + 1]);
}

void update(int node, int lo, int hi, int pos, long long val) {
    if (lo == hi) { tree[node] = leaf(val); return; }
    int mid = (lo + hi) / 2;
    if (pos <= mid) update(node * 2, lo, mid, pos, val);
    else update(node * 2 + 1, mid + 1, hi, pos, val);
    tree[node] = merge(tree[node * 2], tree[node * 2 + 1]);
}

Node query(int node, int lo, int hi, int l, int r) {
    if (r < lo || hi < l) return identity();
    if (l <= lo && hi <= r) return tree[node];
    int mid = (lo + hi) / 2;
    if (r <= mid) return query(node * 2, lo, mid, l, r);
    if (l > mid)  return query(node * 2 + 1, mid + 1, hi, l, r);
    return merge(query(node * 2, lo, mid, l, r),
                 query(node * 2 + 1, mid + 1, hi, l, r));
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    if (!(cin >> n >> q)) return 0;
    a.resize(n);
    for (auto &x : a) cin >> x;

    tree.assign(4 * max(n, 1), identity());
    if (n > 0) build(1, 0, n - 1);

    string out;
    for (int k = 0; k < q; k++) {
        int type;
        cin >> type;
        if (type == 1) {
            int i; long long x;
            cin >> i >> x;
            update(1, 0, n - 1, i, x);
        } else {
            int l, r;
            cin >> l >> r;
            Node res = query(1, 0, n - 1, l, r);
            out += to_string(res.best);
            out += '\n';
        }
    }
    cout << out;
    return 0;
}
```
