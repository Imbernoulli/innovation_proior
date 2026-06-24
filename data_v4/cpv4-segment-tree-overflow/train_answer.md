**Problem.** Maintain `n` integer balances `a[1..n]` (may be negative) under `q` operations: type `1 l r v` adds `v` to every balance in the inclusive range `[l, r]`, and type `2 l r` asks for the exact sum of balances in `[l, r]`. For each audit, print the sum. With `n, q <= 2*10^5` and `|a[i]|, |v| <= 10^9`, both operations must be sub-linear.

**Key idea — segment tree with lazy propagation.** Store subtree sums in a tree over the array. A range adjustment that fully covers a node's segment is recorded as one lazy "add `v` to every element under me" tag plus an immediate correction to the node's stored sum; the tag is pushed to children only when a later query or partial update must descend past the node. The tag algebra has two pieces:

- *Apply add `v` to a node covering `[lo, hi]`:* the segment sum gains `v` once per element, so `tree[node] += (hi - lo + 1) * v`, and `lazy[node] += v` (tags compose additively).
- *Push down:* hand the parent's tag to each child **over the child's own segment** — `[lo, mid]` for the left, `[mid+1, hi]` for the right — then clear the parent's tag.

Update and query use the standard three-case overlap test (disjoint / fully contained / partial); on a partial update, recombine `tree[node] = tree[left] + tree[right]` after recursing. Each operation is `O(log n)`, the whole run `O((n + q) log n)`.

**Pitfalls.**
1. *Int overflow — the headline trap.* The audited sums grow far past 32 bits. Even building the node over three `10^9` balances yields `3*10^9`, which wraps a signed 32-bit `int` to a negative garbage value; the trivial audit `2 1 5` on five `10^9` balances should be `5000000000` but an `int` tree prints `705032704`. Every sum-bearing quantity — `tree`, `lazy`, the increment `(long long)(hi - lo + 1) * v`, and the printed value — must be `long long`.
2. *Wrong child segment in push.* `applyAdd` multiplies the tag by the node's element *count*, so the push must pass each child its own half (`[lo, mid]` and `[mid+1, hi]`), not the parent's `[lo, hi]`; passing the parent's range doubles the count factor and corrupts every leaf below. (A trace of audit `[0,0]` after a `+10` returning `21` instead of `11` exposes exactly this.)
3. *Missing recombine.* After a partial update recurses into children, the node's stored sum is stale until you reset `tree[node] = tree[left] + tree[right]`.

**Edge cases.** `n = 1` (single leaf, no children to push to); whole-array `1 n` (root fully contained, fast path, no descent); single-element query `l == r` (forces a full push to a leaf); `v = 0` (a genuine no-op — `applyAdd` adds `0`, `push` skips a zero tag); negative balances and negative `v` (all quantities signed `long long`).

**Complexity.** `O((n + q) log n)` time, `O(n)` space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int n, q;
vector<long long> tree;   // subtree sums
vector<long long> lazy;   // pending per-element add to push down

void build(const vector<long long> &a, int node, int lo, int hi) {
    if (lo == hi) { tree[node] = a[lo]; return; }
    int mid = (lo + hi) / 2;
    build(a, 2 * node, lo, mid);
    build(a, 2 * node + 1, mid + 1, hi);
    tree[node] = tree[2 * node] + tree[2 * node + 1];
}

// apply an "add v to every element of this node's range" to the node aggregate + its lazy
void applyAdd(int node, int lo, int hi, long long v) {
    tree[node] += (long long)(hi - lo + 1) * v;   // count of elements times v
    lazy[node] += v;
}

void push(int node, int lo, int hi) {
    if (lazy[node] != 0) {
        int mid = (lo + hi) / 2;
        applyAdd(2 * node, lo, mid, lazy[node]);
        applyAdd(2 * node + 1, mid + 1, hi, lazy[node]);
        lazy[node] = 0;
    }
}

// add v to all elements in [ql, qr]
void update(int node, int lo, int hi, int ql, int qr, long long v) {
    if (qr < lo || hi < ql) return;
    if (ql <= lo && hi <= qr) { applyAdd(node, lo, hi, v); return; }
    push(node, lo, hi);
    int mid = (lo + hi) / 2;
    update(2 * node, lo, mid, ql, qr, v);
    update(2 * node + 1, mid + 1, hi, ql, qr, v);
    tree[node] = tree[2 * node] + tree[2 * node + 1];
}

// sum of elements in [ql, qr]
long long query(int node, int lo, int hi, int ql, int qr) {
    if (qr < lo || hi < ql) return 0;
    if (ql <= lo && hi <= qr) return tree[node];
    push(node, lo, hi);
    int mid = (lo + hi) / 2;
    return query(2 * node, lo, mid, ql, qr) + query(2 * node + 1, mid + 1, hi, ql, qr);
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    if (!(cin >> n >> q)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    tree.assign(4 * n, 0);
    lazy.assign(4 * n, 0);
    if (n > 0) build(a, 1, 0, n - 1);

    while (q--) {
        int type;
        cin >> type;
        if (type == 1) {
            int l, r;
            long long v;
            cin >> l >> r >> v;             // 1-indexed inclusive range, add v
            update(1, 0, n - 1, l - 1, r - 1, v);
        } else {
            int l, r;
            cin >> l >> r;                  // 1-indexed inclusive range, query sum
            cout << query(1, 0, n - 1, l - 1, r - 1) << "\n";
        }
    }
    return 0;
}
```
