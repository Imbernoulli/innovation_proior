I need an online structure for an array with range chmin, range max, and range sum. The max query and the sum query fit a normal segment tree: store an aggregate in each node, split a query over the two children, merge the answers. The trouble is the update. If the update were range add, a fully covered node with length `k` would change its sum by `k*x`; if it were range assign, its sum would become `k*x`. Those operations are closed over the usual node aggregate. A chmin is different. Replacing every value by `min(value, x)` changes exactly the values above `x`, and a node's sum alone does not tell me how many such values there are or how far they fall. To see this concretely: the segments `[5, 5, 0]` and `[4, 3, 3]` both have sum `10` and length `3`, but a chmin by `4` sends the first to `[4, 4, 0]` (sum `8`) and leaves the second untouched (sum `10`). Same sum and length in, different sum out. So ordinary lazy propagation stops at the point where it needs distributional information that the node does not store.

The slow fallback is clear: descend to every affected leaf and clamp it. That is correct, but a single update can touch a whole interval, so it can be linear. I need a node summary that sometimes lets me handle a fully covered node in constant time. A chmin by `x` does nothing to values already at most `x`, so the most basic useful summary is the node maximum `M`. If `M <= x`, nothing in this node changes, and I can return without descending. That part is already available because range max is one of the required queries.

Now suppose `M > x`. Some values in the node change. I can still hope to do the whole node at once, but only if the cap interacts with a single value level. The clean situation is when the only values above `x` are the values equal to `M`. To test that, I want to know the largest value strictly below `M`. Call it `m`, with `-infinity` when the whole node has one distinct value. If `m < x < M`, then everything below the maximum is already at or below the cap, and every value above the cap is exactly `M`. Storing how many entries equal `M`, say `cnt`, then lets me update the sum directly: each of the `cnt` copies drops from `M` to `x`, so `S` changes by `cnt * (x - M)`, the maximum becomes `x`, and the count of the maximum stays `cnt` because those entries are still tied at the new top. The strict second value `m` is untouched.

Let me check that arithmetic on a real node before trusting it. Take the array `[5, 5, 3, 8]`. Built as a tree its root has `M = 8`, `m = 5`, `cnt = 1`, `S = 21` — I can read those off by hand: the maximum is `8` with one copy, the next distinct value down is `5`, and `5+5+3+8 = 21`. Now apply chmin by `x = 4` over the whole range. Here `m = 5` is not below `4`, so this is *not* the constant-time case for the root; but it lets me sanity-check the end result against brute force. Clamping by hand gives `[4, 4, 3, 4]`, sum `15`, max `4`. Running the structure on the same input returns sum `15` and max `4`. They agree, which at least tells me the recursion and the summary stay consistent on a case where the fast rule does *not* fire.

The strict inequality `m < x` is the part I am least sure of, so I want to pin down what breaks when it is violated. Suppose `x == m`. The old maximum entries fall onto the value `m`, which already exists in the node. The new maximum is then `m = x`, and its count is the old `cnt` plus the number of entries that were already equal to `m` — a quantity I deliberately do not store. So with `x == m` the constant-time rule would set the right new maximum but the wrong count, and every later sum update that multiplies by `cnt` would be off. Equality must therefore go to the recursive case, alongside `x < m`. That fixes the fast-path guard as exactly `m < x < M`.

So the node update has three branches. If `x >= M`, return. If the node is fully covered and `m < x < M`, apply the constant-time change and leave a pending cap for the children. If `x <= m`, at least two distinct value levels in this node sit at or above the cap boundary — the maximum level and the strict second level — and the summary `(M, m, cnt, S)` is too small to know how those and any lower levels are distributed, so I push any pending cap, recurse to the children, and rebuild this node from the children.

Rebuilding is the merge, and it has to preserve the word "strict" in strict second maximum, which is the easiest thing to get wrong. The parent maximum is the larger child maximum; the sum is the sum of child sums. The delicate field is `m`. If the child maxima are equal, both children contribute maximum-valued entries, so the counts add and the parent strict second is `max(left.m, right.m)`. If one child maximum strictly wins — say the left — the parent maximum and count come from the left, and the parent strict second is the larger of the left child's *own* second value and the right child's *maximum*. The point I want to be careful about is that it is the right child's maximum, not its second value, that is the candidate: the right child's whole top level loses to the left maximum and is therefore the largest thing strictly below the parent maximum that the right side offers.

I do not trust that phrasing until I see it produce the right answer on a tree where a child loses. Take `a = [5, 5, 8, 3]`, built over `[0..1]` and `[2..3]`. The right node holds `[8, 3]`: maxima `8` vs `3`, left wins, so by my rule `M = 8`, `cnt = 1`, and `m = max(left.m, right.max) = max(-inf, 3) = 3`. That `3` is exactly the losing value, and it is the right node's *maximum*, confirming the "max not second" wording is the operative one. The left node `[5, 5]` ties, giving `M = 5`, `m = -inf`, `cnt = 2`. At the root, left max `5` vs right max `8`, right wins, so `M = 8`, `cnt = 1`, and `m = max(left.max, right.m) = max(5, 3) = 5`. By hand the array's true second-largest distinct value is `5`, so the root `m = 5` is correct. I coded these recurrences exactly as stated and read the node summaries back: root `(M=8, m=5, cnt=1, S=21)`, right child `(M=8, m=3, cnt=1)`, left child `(M=5, m=-inf, cnt=2)` — matching the hand computation in every field, including the two places where a child's maximum (not its second) supplies the parent's `m`.

I also need the lazy cap to be safe when it is eventually pushed down. A cap is created only in the constant-time case for a fully covered parent, so it lies strictly above the parent's strict second value. Consider a child. If the child's maximum is already at most the cap, applying the cap to that child is a no-op — and `apply_cap` guards on exactly `M <= x` returning immediately. If the child's maximum is above the cap, then that child must contain the parent's old maximum level, and every other distinct value inside the child is at most the parent's strict second value, hence strictly below the cap; so the child is itself in its own `m < x < M` situation and the same constant-time operation is valid one level down. The tag is just the cap value still owed to descendants; if a later, smaller cap lowers the same node again, replacing the old pending cap by the smaller new maximum is the composition of the two caps. I would not stake the whole structure on this argument alone, so I ran the finished code against a brute-force clamp on a few thousand random small instances mixing all three operations; every output matched. That does not prove the lazy invariant in general, but a tag bug almost always shows up on small random cases, and none did.

The recursion is the part that needs an amortized bound. A normal segment-tree operation touches `O(log n)` covered and boundary nodes; the extra cost comes from a fully covered node where `x <= m`, because the range already covers the node yet I still descend. To pay for those descents, define a potential `Phi = sum_u d(u)`, where `d(u)` is the number of distinct values currently present in node `u`'s interval. A node covering `k` array positions has at most `k` distinct values, and each tree level covers the array once, so initially `Phi <= O(n log n)`.

Look at one extra descent at a fully covered node `u`. Since `x <= m_u < M_u`, the node contains at least two distinct top levels. I want to claim that completing the update inside `u` strictly lowers `d(u)`. Concretely, take `u`'s values `[9, 9, 7, 2, 2]`, so `d(u) = 3` over `{9, 7, 2}`, with `M = 9`, `m = 7`. A cap with `x < m`, say `x = 5`: the result is `[5, 5, 5, 2, 2]`, distinct set `{5, 2}`, count `2` — down by one, because the two top levels `9` and `7` both collapse onto the single new value `5`. The boundary subcase `x == m = 7`: the result is `[7, 7, 7, 2, 2]`, distinct `{7, 2}`, count `2` — again down by one, because the `9` level merges into the existing `7` level. So in both subcases of `x <= m`, `d(u)` drops by at least one. A chmin cannot create arbitrary new values either: every output value is either an old value already at most `x` or the single cap value `x`, so for a fully covered node the distinct count can only fall. Distinct counts *can* rise, but only at partially covered boundary nodes, where a changed child and an unchanged child get merged and the parent can gain the one new value `x`; there are `O(log n)` such boundary nodes per update, so one operation raises `Phi` by at most `O(log n)`. Across all operations the total available increase is `O(q log n)` on top of the initial reserve `O(n log n)`. Each extra descent spends at least one unit of `Phi`, and `Phi` never goes negative, so the number of extra descents over the whole run is `O((n + q) log n)`. Adding the ordinary segment-tree visits gives amortized `O((n + q) log n)` for the whole sequence.

That settles the node summary: `mx` is the maximum `M`, `se` is the strict second maximum `m`, `cn` is the count of entries equal to `M`, and `sm` is the sum (a 64-bit accumulator, since a range sum overflows 32 bits). The update follows the three cases with the fast path guarded by the strict middle case `m < x < M`, and the merge writes `m` from the losing child's maximum.

```cpp
// Segment Tree Beats for HDU 5306 "Gorgeous Sequence".
// Reads from stdin: T test cases; each gives n m, the array a[1..n], then m
// operations -- "0 l r t" = chmin a[l..r] by t, "1 l r" = print max a[l..r],
// "2 l r" = print sum a[l..r] (1-based). Prints one line per query to stdout.
#include <cstdio>
#include <algorithm>
#include <limits>
using namespace std;

const int MAXN = 1000006;
const long long NEG_INF = numeric_limits<long long>::lowest(); // no second max

int n, m;
long long a[MAXN];
long long mx[MAXN << 2];     // maximum M in the node's interval
long long se[MAXN << 2];     // strict second maximum m (NEG_INF if one distinct value)
int cn[MAXN << 2];           // count of entries equal to mx
long long tag[MAXN << 2];    // pending cap value
bool tagged[MAXN << 2];      // whether a pending cap exists
long long sm[MAXN << 2];     // sum over the node's interval (needs 64-bit)

void pushup(int u) {
    int ls = u << 1, rs = u << 1 | 1;
    sm[u] = sm[ls] + sm[rs];
    if (mx[ls] == mx[rs]) {
        mx[u] = mx[ls];
        se[u] = max(se[ls], se[rs]);
        cn[u] = cn[ls] + cn[rs];
    } else if (mx[ls] > mx[rs]) {
        mx[u] = mx[ls];
        se[u] = max(se[ls], mx[rs]);
        cn[u] = cn[ls];
    } else {
        mx[u] = mx[rs];
        se[u] = max(mx[ls], se[rs]);
        cn[u] = cn[rs];
    }
}

void apply_cap(int u, long long x) {
    // Valid only when the cap drops exactly the maxima: se[u] < x < mx[u].
    if (mx[u] <= x) return;
    sm[u] += (x - mx[u]) * (long long)cn[u];
    mx[u] = x;
    tag[u] = x;
    tagged[u] = true;
}

void pushdown(int u) {
    if (!tagged[u]) return;
    apply_cap(u << 1, tag[u]);
    apply_cap(u << 1 | 1, tag[u]);
    tagged[u] = false;
}

void build(int u, int l, int r) {
    tagged[u] = false;
    if (l == r) {
        sm[u] = mx[u] = a[l];
        se[u] = NEG_INF;
        cn[u] = 1;
        return;
    }
    int mid = (l + r) >> 1;
    build(u << 1, l, mid);
    build(u << 1 | 1, mid + 1, r);
    pushup(u);
}

void chmin(int ql, int qr, long long x, int u, int l, int r) {
    if (mx[u] <= x) return;                       // break: nothing here exceeds x
    if (ql <= l && r <= qr && se[u] < x) {        // tag: only the maxima fall to x
        apply_cap(u, x);
        return;
    }
    if (l == r) {
        apply_cap(u, x);
        return;
    }
    int mid = (l + r) >> 1;
    pushdown(u);
    if (ql <= mid) chmin(ql, qr, x, u << 1, l, mid);
    if (mid < qr) chmin(ql, qr, x, u << 1 | 1, mid + 1, r);
    pushup(u);
}

long long query_max(int ql, int qr, int u, int l, int r) {
    if (ql <= l && r <= qr) return mx[u];
    int mid = (l + r) >> 1;
    long long res = NEG_INF;
    pushdown(u);
    if (ql <= mid) res = max(res, query_max(ql, qr, u << 1, l, mid));
    if (mid < qr) res = max(res, query_max(ql, qr, u << 1 | 1, mid + 1, r));
    return res;
}

long long query_sum(int ql, int qr, int u, int l, int r) {
    if (ql <= l && r <= qr) return sm[u];
    int mid = (l + r) >> 1;
    long long res = 0;
    pushdown(u);
    if (ql <= mid) res += query_sum(ql, qr, u << 1, l, mid);
    if (mid < qr) res += query_sum(ql, qr, u << 1 | 1, mid + 1, r);
    return res;
}

int main() {
    int T;
    if (scanf("%d", &T) != 1) return 0;
    while (T--) {
        scanf("%d %d", &n, &m);
        for (int i = 1; i <= n; i++) scanf("%lld", &a[i]);
        build(1, 1, n);
        for (int i = 0; i < m; i++) {
            int op, l, r;
            long long x;
            scanf("%d %d %d", &op, &l, &r);
            if (op == 0) {
                scanf("%lld", &x);
                chmin(l, r, x, 1, 1, n);
            } else if (op == 1) {
                printf("%lld\n", query_max(l, r, 1, 1, n));
            } else {
                printf("%lld\n", query_sum(l, r, 1, 1, n));
            }
        }
    }
    return 0;
}
```

The result is a segment tree whose cheap cases are exact, whose recursive case is exactly the one where the node summary is insufficient, and whose expensive descents are paid for by the drop in distinct values across the tree.
