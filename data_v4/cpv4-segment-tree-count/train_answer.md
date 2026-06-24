**Problem.** Heights `h[0..n-1]`. Support `1 p x` (set `h[p] = x`) and `2 l r` (report how many buildings are *visible* in window `[l, r]`, where a building is visible iff it is strictly taller than every building to its left inside the window). The query count equals the number of **strict prefix maxima** of `h[l..r]`. Read everything from stdin, print one line per query. `n, q <= 2*10^5`, `1 <= h[i] <= 10^9`.

**Why the obvious approach is too slow.** Recomputing the running maximum across `h[l..r]` is `O(window)` per query and obviously correct (this is the brute oracle), but a window can span the whole array, so `q` queries are `O(n*q) = 4*10^{10}`. Too slow.

**Key idea — max segment tree with a threshold descent, threaded left to right.** Keep a max segment tree over `h`; point update is the textbook `O(log n)`. For a query, define `countVisible(node, bound)` = number of strict prefix maxima inside the node *given* that everything seen to its left has maximum `bound`:

- if `mx[node] <= bound`, prune (return `0`): nothing here can strictly exceed the running max;
- at a leaf with `mx > bound`, return `1`;
- otherwise recurse left with `bound`, then recurse right with `max(bound, mx[left])`, and add. This is `O(log n)` per node because dominated subtrees are pruned.

For the full window, walk its `O(log n)` canonical segments **left to right** (recurse left child before right child), carrying a running `bound`: count inside each fully covered node with the current `bound`, then raise `bound` by that node's max before the next segment. Total `O(log^2 n)` per query.

**Pitfalls.**
1. *Double-count across segments (the headline bug).* Visibility is defined relative to *everything to the left in the query*, so it is **not** additive per node. Counting each canonical segment independently (with `bound` reset to `-inf`) over-counts: a later segment's elements get credited as local prefix maxima even though a taller earlier segment already dominates them. On `[1,1,1,9,2,3,4,5]` query `[0,7]` this returns `6` instead of `2` (the `9` should suppress `2,3,4,5`). Fix: thread a single running `bound` through the query, raising it by each fully covered node's max, and visit segments in array order (left child first).
2. *Strict vs non-strict (off-by-one on ties).* Visibility is strict `>`. The prune must be `mx[node] <= bound`: a node whose max merely *ties* the bound contributes nothing. Using `<` counts equal heights as visible — on `[4,4,4]` it returns `3` instead of `1`. Only the first of a run of equal heights is visible.

**Edge cases.** `n = 1` or `l == r` -> `1` (the leftmost/only building is always visible). All-equal window -> `1`. Strictly increasing window -> window length (every building visible). Strictly decreasing -> `1`. Heights up to `10^9` are only compared, never summed, so no overflow; `long long` is used throughout for comfort.

**Complexity.** `O(log n)` per update, `O(log^2 n)` per query, `O(n)` memory. At `n = q = 2*10^5` this runs in well under a second.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int n, q;
vector<long long> h;       // current heights
vector<long long> mx;      // segment-tree node maxima

void build(int node, int lo, int hi) {
    if (lo == hi) { mx[node] = h[lo]; return; }
    int mid = (lo + hi) / 2;
    build(2 * node, lo, mid);
    build(2 * node + 1, mid + 1, hi);
    mx[node] = max(mx[2 * node], mx[2 * node + 1]);
}

void pointSet(int node, int lo, int hi, int pos, long long val) {
    if (lo == hi) { mx[node] = val; return; }
    int mid = (lo + hi) / 2;
    if (pos <= mid) pointSet(2 * node, lo, mid, pos, val);
    else            pointSet(2 * node + 1, mid + 1, hi, pos, val);
    mx[node] = max(mx[2 * node], mx[2 * node + 1]);
}

// Within this node's range, count positions that are STRICTLY greater than
// `bound` AND than everything to their left inside the node. Equivalent to:
// number of strict prefix maxima of the node's segment when the running max
// starts at `bound`. O(log n) because we prune any subtree whose max <= bound.
long long countVisible(int node, int lo, int hi, long long bound) {
    if (mx[node] <= bound) return 0;          // whole subtree is dominated
    if (lo == hi) return 1;                    // single element, mx > bound
    int mid = (lo + hi) / 2;
    long long leftCnt = countVisible(2 * node, lo, mid, bound);
    // For the right half the bound is raised by the left half's max: any new
    // prefix maximum in the right half must beat both the external bound and
    // everything in the left half.
    long long newBound = max(bound, mx[2 * node]);
    long long rightCnt = countVisible(2 * node + 1, mid + 1, hi, newBound);
    return leftCnt + rightCnt;
}

// Walk the query range [ql,qr] left to right over the O(log n) canonical
// segments. `bound` is the running prefix maximum accumulated from the query
// segments already consumed to the LEFT; after counting inside a fully covered
// node we raise `bound` by that node's max so the next segment is gated.
long long queryCount(int node, int lo, int hi, int ql, int qr, long long &bound) {
    if (ql <= lo && hi <= qr) {
        long long c = countVisible(node, lo, hi, bound);
        bound = max(bound, mx[node]);
        return c;
    }
    int mid = (lo + hi) / 2;
    long long res = 0;
    if (ql <= mid)  res += queryCount(2 * node, lo, mid, ql, qr, bound);
    if (qr > mid)   res += queryCount(2 * node + 1, mid + 1, hi, ql, qr, bound);
    return res;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    if (!(cin >> n >> q)) return 0;
    h.assign(max(1, n), 0);
    for (int i = 0; i < n; i++) cin >> h[i];
    mx.assign(4 * max(1, n), LLONG_MIN);
    if (n > 0) build(1, 0, n - 1);

    string out;
    for (int i = 0; i < q; i++) {
        int type;
        cin >> type;
        if (type == 1) {
            int p; long long x;
            cin >> p >> x;
            h[p] = x;
            pointSet(1, 0, n - 1, p, x);
        } else {
            int l, r;
            cin >> l >> r;
            long long bound = LLONG_MIN;           // nothing to the left yet
            long long c = queryCount(1, 0, n - 1, l, r, bound);
            out += to_string(c);
            out += '\n';
        }
    }
    cout << out;
    return 0;
}
```
