We need an online structure for an integer array $a[1..n]$ under three interleaved operations: a range cap $\texttt{chmin}(l,r,x)$ that sets $a[i] \leftarrow \min(a[i], x)$ for all $i \in [l,r]$, a range maximum query, and a range sum query. Both the array and the operation stream can be huge, so re-scanning a range per operation is too slow. The two read-only queries are already comfortable for a segment tree: store an aggregate per node and merge the two children's answers when a query splits. The hard part is the update. If the update were a range add, a fully covered node of length $k$ would change its sum by $k\cdot x$; if it were a range assign, its sum would become $k\cdot x$. Both are closed over the ordinary sum/max summary, so lazy propagation handles them with a single tag. A $\texttt{chmin}$ is different in kind: $\min(a_i, x)$ touches exactly the entries above $x$ and leaves the rest alone, and a node's sum and maximum do not reveal how many entries exceed $x$ or how far they fall. Two segments can share length, maximum, and sum yet respond differently to the same cap. So ordinary lazy propagation stalls precisely at the point where it would need distributional information the node never stored, and the only correct fallback — descending to every affected leaf and clamping it one by one — is linear per update.

The method is Segment Tree Beats. The idea is to enrich each node's summary by exactly the one extra fact a cap needs, so that most fully covered nodes can still be updated in constant time, and to descend only in the genuinely insufficient case — then pay for those descents amortized. Each node over its interval stores four quantities: the maximum $M$, the strict second maximum $m$ (the largest value strictly below $M$, taken as $-\infty$ when the interval holds a single distinct value), the count $cnt$ of entries equal to $M$, and the sum $S$. A cap by $x$ at a node then splits cleanly into three cases. If $M \le x$, nothing in the node exceeds the cap and we return immediately — this case reuses the maximum we already keep for the range-max query. If $m < x < M$, then every value strictly below the maximum is already at or below $x$, so the only entries above the cap are the $cnt$ copies of $M$; the entire node updates in constant time by
$$S \mathrel{+}= cnt\,(x - M), \qquad M \leftarrow x,$$
with $m$ and $cnt$ unchanged, and a pending cap left for the descendants. The strict inequality $x > m$ is load-bearing and not a convenience: if $x = m$, the old maximum entries collapse onto the already-present level $m$, so the new count of the maximum would be $cnt$ plus the number of old-$m$ entries, a quantity the summary does not record. Equality must therefore fall to the recursive case. That leaves the third case, $x \le m$: now at least two distinct levels — the maximum and the strict second maximum — sit at or above the cap boundary, the four-number summary is deliberately too coarse to know how they redistribute, so we push any pending cap down, recurse into both children, and rebuild the node from them.

The merge that rebuilds a parent must preserve the word "strict" in the strict second maximum, and this is where the three branches of $\texttt{pushup}$ come from. The parent maximum is the larger of the two child maxima. When the child maxima tie, both children supply maximum-valued entries, so their counts add and the parent's strict second is $\max$ of the two child strict-seconds. When the left maximum strictly wins, the parent maximum and count come from the left child, and the parent's strict second is $\max$ of the left child's own strict second and the *right child's maximum* — it is the right maximum, not the right strict second, that is the largest value sitting just below the parent maximum. The right-wins case is symmetric, and the sum is always the sum of the children. The lazy cap is safe for the same reason the constant-time case is exact: a tag is created only when capping a fully covered node in the strict middle case, so the cap value lies strictly above that node's strict second maximum. Pushed to a child, either the child's maximum is already $\le x$ and the cap is a no-op, or the child's maximum exceeds $x$, in which case that child holds the parent's old maximum level while all its other distinct values are at most the parent's strict second, hence strictly below the cap — so the very same constant-time apply is valid there. A tag is just the cap still owed to descendants, and overwriting an old pending cap by a smaller new one is exactly the composition $\min(\cdot, x_{\text{old}}) \circ \min(\cdot, x_{\text{new}})$ when $x_{\text{new}} \le x_{\text{old}}$.

What makes this fast is an amortized argument over the whole operation sequence rather than per update, because a single $\texttt{chmin}$ can still open several nodes. Define the potential $\Phi = \sum_u d(u)$, where $d(u)$ is the number of distinct values currently in node $u$'s interval. A node covering $k$ positions has at most $k$ distinct values, and each tree level covers the array once, so initially $\Phi = O(n\log n)$. Consider one extra descent — a fully covered node $u$ with $x \le m_u < M_u$, where the range already covers the node yet we still recurse. Because $x \le m_u$, the node has at least two distinct top levels; if $x = m_u$ the maximum level merges onto the second level, and if $x < m_u$ both top levels (and possibly more) collapse onto the single new value $x$. Either way, once the update inside $u$ completes, $d(u)$ has dropped by at least one. A cap never manufactures arbitrary new values — every resulting value is either an old value that was already $\le x$ or the single cap value $x$ — so for a fully covered node the distinct count cannot rise. Distinct counts can only increase at the $O(\log n)$ partially covered boundary nodes of an update, where capped and uncapped children are merged, and each such node gains at most the one new value $x$. Hence one operation raises $\Phi$ by at most $O(\log n)$; over $q$ operations the total injectable increase is $O(q\log n)$ on top of the $O(n\log n)$ initial reserve. Since each extra descent spends at least one unit of $\Phi$ and $\Phi \ge 0$ always, the extra descents total $O((n+q)\log n)$, and adding the ordinary segment-tree visits gives an overall amortized $O((n+q)\log n)$ for the entire sequence of updates and queries. The concrete summary therefore maps directly to the storage — $\texttt{mx}$ is $M$, $\texttt{se}$ is $m$, $\texttt{cn}$ is $cnt$, $\texttt{sm}$ is $S$, and $\texttt{tag}$ is the pending cap — and the update simply executes the three cases without ever weakening the strict middle inequality. The array sum uses a 64-bit accumulator since per-element values up to the input bound summed over a range overflow a 32-bit integer.

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
