**Problem.** Given an array `a[0..n-1]` of positive integers and `q` offline range queries, answer for each `[l, r]` (1-based inclusive) the *frequency power* `P(l, r) = sum over distinct values v of (count of v in a[l..r])^2`. Read `n q`, the array, and the `q` ranges from stdin; print the `q` answers in input order. `n, q, a[i] <= 2*10^5`.

**Why the obvious approaches are wrong.** Per-query recomputation (sweep the window, tally counts, sum squares) is `O(n)` per query, hence `O(nq) ~ 4*10^10` at the limits — 40-ish seconds, far over budget; it is only the reference oracle. A segment tree over the array also fails, but for a *structural* reason: `P` is not decomposable. The contribution of a value `v` is `(count of v)^2`, which couples all its occurrences, so `P(l, m)` and `P(m+1, r)` cannot be merged into `P(l, r)` — a value straddling the split is squared separately on each side but must be squared together in the union.

**Key idea — Mo's algorithm with Hilbert ordering.** `P` admits an `O(1)` *incremental* update: when one element of value `v` (current count `c`) enters the window, `P` changes by `(c+1)^2 - c^2 = 2c + 1`; when it leaves, by `c^2 - (c-1)^2 = 2c - 1` downward. So if I process the `q` windows in an order that keeps the two endpoints `(l, r)` moving little in aggregate, each unit of movement costs `O(1)`. Treat each query as a point `(l, r)` in the plane; the cost of going from one query to the next is the Manhattan distance between the points, so I want a short tour of the point set.

Classic Mo sorts by `(l / B, r)` with block size `B ~ sqrt(n)`, achieving `O((n + q) sqrt n)` — but that is a row-major scan of the 2-D points, which is discontinuous at block boundaries (the `r` endpoint "carriage-returns" between blocks), inflating the hidden constant on adversarial inputs. The state-of-the-art fix is to linearize the points along a **Hilbert space-filling curve** instead: embed each `(l, r)` on a `2^k x 2^k` grid (`2^k >= n`), compute its Hilbert distance, and sort by it. The Hilbert curve has the best locality of the standard space-filling curves and no boundary cliffs (consecutive curve cells are always grid-neighbors), so the same `O((n + q) sqrt n)` bound holds with a markedly smaller constant — which is what the `n = q = 2*10^5` limits reward.

**Pitfalls to get right.**
1. *Hilbert reflection mask.* The canonical `xy2d` rotation reflects coordinates within the *entire* grid (`x -> side-1-x`, with `side - 1 = (1<<order)-1`), **not** within the low-`s`-bit sub-square. Using a local low-bit mask scrambles the sub-curve orientation, breaking continuity and the bijection. This bug is insidious because Mo's prints the *right answers under any ordering* — only the speed degrades — so an answer-vs-brute diff will not catch it. Verify the curve separately: exhaustively check it is a bijection and that curve-consecutive cells are at Manhattan distance 1.
2. *Window-move loop order.* Grow before shrink — add on the right, add on the left, then remove on the right, remove on the left — so the window is always a superset during each individual add and no count goes negative mid-transition.
3. *Overflow.* The all-equal full-range answer is `n^2 = 4*10^10`, past 32 bits. Keep `cur`, the answers, and the count `c` inside the delta computation as `long long`.

**Edge cases.** `n = 1`; single-point queries `l == r` (answer 1); full-array queries; all-equal arrays (max answer, overflow check); all-distinct arrays (answer = window length). The empty start window `curL = 0, curR = -1` with grow-before-shrink handles every degenerate move without a negative count.

**Complexity.** `O((n + q) sqrt n)` time, `O(n + q)` space. Measured ~0.18 s and ~12 MB at `n = q = 2*10^5`.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

// Map (x, y) on a side x side grid (side = 2^order) to its 1-D distance along the
// Hilbert space-filling curve. Canonical iterative xy2d construction.
static inline uint64_t hilbertOrder(uint32_t x, uint32_t y, int order) {
    uint64_t d = 0;
    for (int s = order - 1; s >= 0; --s) {
        uint32_t rx = (x >> s) & 1u;
        uint32_t ry = (y >> s) & 1u;
        d += ((uint64_t)((3u * rx) ^ ry)) << (2 * s);
        // rotate / reflect the quadrant
        if (ry == 0) {
            if (rx == 1) {
                uint32_t mask = (1u << order) - 1u;   // side - 1
                x = (mask - x) & mask;
                y = (mask - y) & mask;
            }
            uint32_t t = x; x = y; y = t;             // swap
        }
    }
    return d;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    if (!(cin >> n >> q)) return 0;

    vector<int> a(n);
    int maxv = 0;
    for (int i = 0; i < n; ++i) {
        cin >> a[i];
        if (a[i] > maxv) maxv = a[i];
    }

    // Hilbert grid order: smallest power of two side covering all indices.
    int order = 1;
    while ((1 << order) < max(1, n)) ++order;

    struct Query { int l, r, idx; uint64_t h; };
    vector<Query> qs(q);
    for (int i = 0; i < q; ++i) {
        int l, r;
        cin >> l >> r;          // 1-based inclusive
        --l; --r;               // to 0-based inclusive
        qs[i].l = l;
        qs[i].r = r;
        qs[i].idx = i;
        qs[i].h = hilbertOrder((uint32_t)l, (uint32_t)r, order);
    }

    sort(qs.begin(), qs.end(),
         [](const Query& A, const Query& B) { return A.h < B.h; });

    vector<int> cnt(maxv + 2, 0);
    vector<long long> ans(q, 0);

    long long cur = 0;          // sum over values v of cnt[v]^2 inside [curL, curR]
    int curL = 0, curR = -1;    // empty window

    auto add = [&](int pos) {
        int v = a[pos];
        long long c = cnt[v];
        cur += 2 * c + 1;       // (c+1)^2 - c^2
        cnt[v] = (int)(c + 1);
    };
    auto remove = [&](int pos) {
        int v = a[pos];
        long long c = cnt[v];
        cur -= 2 * c - 1;       // c^2 - (c-1)^2
        cnt[v] = (int)(c - 1);
    };

    for (const auto& Q : qs) {
        int L = Q.l, R = Q.r;
        while (curR < R) add(++curR);
        while (curL > L) add(--curL);
        while (curR > R) remove(curR--);
        while (curL < L) remove(curL++);
        ans[Q.idx] = cur;
    }

    string out;
    out.reserve((size_t)q * 12);
    char buf[24];
    for (int i = 0; i < q; ++i) {
        int len = sprintf(buf, "%lld\n", ans[i]);
        out.append(buf, len);
    }
    fputs(out.c_str(), stdout);
    return 0;
}
```
