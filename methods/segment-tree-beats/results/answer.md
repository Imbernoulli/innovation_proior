# Segment Tree Beats for Range Chmin, Max, and Sum

Maintain a segment tree whose node summary is strong enough to sometimes apply a
range `chmin` to a fully covered node without opening it. Each node stores:

- `M`: the maximum value in the node interval;
- `m`: the strict second maximum, or `-infinity` if the interval has one distinct value;
- `cnt`: the number of entries equal to `M`;
- `S`: the interval sum.

For a chmin by `x`, the node has three cases. If `M <= x`, no value changes. If
`m < x < M`, the only values above `x` are the `cnt` copies of `M`, so the node is
updated in constant time: `S += cnt * (x - M)` and `M = x`. The strict inequality
is essential: when `x == m`, the old maximum falls onto an existing level and the
new maximum count is not known from this summary, so the update must recurse. If
`x <= m`, recurse into the children and recompute the node.

The merge rule is exact: when child maxima tie, add their counts and take the
larger child second maximum; when one child maximum wins, the losing child's
maximum is a candidate for the parent's strict second maximum. The amortized cost
is `O((n + q) log n)`: every extra descent past a fully covered node with
`x <= m` drops that node's distinct-value count by at least one after the update
inside it completes, while each update can introduce the new cap value only along
`O(log n)` partially covered boundary nodes.

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
