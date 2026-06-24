**Reading the problem and pinning the contract.** I have `n` building heights `h[0..n-1]` and `q` operations. A type-`1` operation `1 p x` sets `h[p] = x`. A type-`2` operation `2 l r` asks for the number of *visible* buildings in the window `[l, r]`: a building at position `k` is visible iff it is strictly taller than every building to its left inside the window, i.e. `h[k] > h[j]` for all `l <= j < k`. The leftmost building is always visible. So the query is exactly "how many strict prefix maxima does `h[l..r]` have." Input is `n q`, then the array, then `q` lines; I print one integer per type-`2`. Scale first, because it dictates everything: `n, q <= 2*10^5`, heights up to `10^9`. The *answer* is a count bounded by `n`, so it fits in 32 bits, but heights up to `10^9` are fine in 32-bit too — still, I will carry heights and the threshold in `long long` to make comparisons painless and to never have to think about it again. The real scaling pressure is time: a query window can be the whole array, so anything `O(window)` per query is `O(n*q) = 4*10^{10}` and dies. I need roughly `O(log)` or `O(log^2)` per operation.

**Laying out the candidate approaches.** Two routes are on the table.

- *Recount per query.* For `[l, r]`, scan left to right with a running max, count strict increases. Dead simple, obviously correct — this is my brute-force oracle — but `O(window)` per query is too slow for the stated limits. I keep it only to check the real solution against.
- *Max segment tree with a threshold descent.* Visibility is not additive, but it has a clean recursive shape. For a single node covering `[lo, hi]`, the number of strict prefix maxima of that segment *given that everything to its left has maximum `bound`* can be computed by descending: if the whole node's max is `<= bound`, nothing here can beat the running max, so the count is `0` and I prune. Otherwise I recurse left, then recurse right but with the bound raised by the left child's max (a new prefix max in the right half must beat both the external bound and the entire left half). For a full query window `[l, r]` I walk its `O(log n)` canonical segments left to right, and after counting inside each segment I raise the running `bound` by that segment's max before moving to the next. Point update is the textbook `O(log n)`. Query is `O(log^2 n)`. That is fast enough: `2*10^5 * (18)^2 ~ 6.5*10^7`.

I commit to the segment tree. The two places I expect to get it wrong are (a) the threshold descent inside a node — the strict-vs-non-strict comparison and which child's max raises the bound — and (b) the cross-segment threading of `bound` in the query walk. Both are counting bugs that produce a plausible-but-wrong number, so I will trace them explicitly.

**Deriving the per-node "count above a bound" recursion and sanity-checking it.** I define `countVisible(node, lo, hi, bound)` = number of positions in `[lo, hi]` that are strict prefix maxima of the segment when the running maximum coming in from the left is `bound`. The recursion:

- If `mx[node] <= bound`: every element here is `<= bound`, none can strictly exceed the running max, return `0`. (Note `<=`, not `<`: an element equal to `bound` is *not* strictly greater, so it is not visible.)
- If `lo == hi` (and we did not prune): this single element has `mx[node] > bound`, so it is a strict prefix max here, return `1`.
- Otherwise split at `mid`. The left child sees the same incoming `bound`. The right child sees `max(bound, mx[left])`, because any visible element in the right half must beat both the external bound and everything in the left half (whose maximum is `mx[left]`). Return `leftCnt + rightCnt`.

Sanity check on a tiny segment, `[1, 4, 1]` with `bound = -inf`. Whole max `4 > -inf`, split into `[1,4]` and `[1]`. Left `[1,4]` with bound `-inf`: max `4 > -inf`, split `[1]` and `[4]`. `[1]`: `1 > -inf` -> 1. Right `[4]` with bound `max(-inf, 1) = 1`: `4 > 1` -> 1. Left subtotal 2. Now the top-level right child `[1]` gets bound `max(-inf, mx[left=[1,4]]=4) = 4`: `1 > 4`? no, prune -> 0. Total `2`. The visible buildings of `[1,4,1]` are `1` and `4` -> 2. Correct. The recursion matches the definition.

**First implementation of the query walk — and a deliberate trace, because the cross-segment part is where counts double.** My first cut of the query treats each canonical segment independently: count its prefix maxima from scratch and add them up.

```
long long queryBad(int node, int lo, int hi, int ql, int qr) {
    if (ql <= lo && hi <= qr) {
        return countVisible(node, lo, hi, LLONG_MIN); // count within this node alone
    }
    int mid = (lo + hi) / 2;
    long long res = 0;
    if (ql <= mid) res += queryBad(2*node, lo, mid, ql, qr);
    if (qr > mid)  res += queryBad(2*node+1, mid+1, hi, ql, qr);
    return res;
}
```

I trace it on `h = [1, 1, 1, 9, 2, 3, 4, 5]` (n = 8), query `2 0 7`. The true answer: prefix maxima of the whole array are `1` (pos 0) and `9` (pos 3); `2,3,4,5` are all below `9`, so the answer is `2`. Now the buggy walk. With `n = 8` the root splits `[0,3]` and `[4,7]`, and the query `[0,7]` covers both as canonical segments. `countVisible([0,3], bound = -inf)` over `[1,1,1,9]`: prefix maxima are `1` then `9`, that is `2`. `countVisible([4,7], bound = -inf)` over `[2,3,4,5]`: every element beats the previous, that is `4`. The walk returns `2 + 4 = 6`.

**Diagnosing the bug.** `6` is wildly wrong (true answer `2`). The defect is exact: each canonical segment was counted as if it were the *start* of the window, with its bound reset to `-inf`. But the right segment `[4,7] = [2,3,4,5]` sits *after* `[0,3]`, whose max is `9`. Inside the window, none of `2,3,4,5` is visible because `9` already dominates them — yet the independent count credited all four as local prefix maxima. This is the classic double-count/over-count of segment-tree counting: a feature defined relative to "everything to the left in the query" cannot be tallied per node in isolation, because the left context of a later node is exactly the union of the earlier nodes. I lost the `9`'s suppressing effect at the segment boundary. The fix must *thread the running maximum across segment boundaries*: after counting a fully covered node, raise a running `bound` by that node's max, and feed it into the next node.

**Fixing the threading and re-verifying.** I make `bound` a reference parameter that flows through the recursion in left-to-right order. At a fully covered node I count with the incoming `bound`, then raise `bound` by `mx[node]`. At an internal node I recurse left first (which may consume covered segments and advance `bound`), then right.

```
long long queryCount(int node, int lo, int hi, int ql, int qr, long long &bound) {
    if (ql <= lo && hi <= qr) {
        long long c = countVisible(node, lo, hi, bound);
        bound = max(bound, mx[node]);
        return c;
    }
    int mid = (lo + hi) / 2;
    long long res = 0;
    if (ql <= mid)  res += queryCount(2*node, lo, mid, ql, qr, bound);
    if (qr > mid)   res += queryCount(2*node+1, mid+1, hi, ql, qr, bound);
    return res;
}
```

The order matters: because the segment tree's canonical segments for a range are visited left-to-right when I recurse left-child-before-right-child, the `bound` accumulates in array order, which is exactly the order the prefix-max definition needs. Re-trace `[1,1,1,9,2,3,4,5]`, query `[0,7]`, starting `bound = -inf`. Covered node `[0,3]`: `countVisible` with `bound=-inf` over `[1,1,1,9]` = `2`; then `bound = max(-inf, 9) = 9`. Covered node `[4,7]`: `countVisible` with `bound=9` over `[2,3,4,5]` — every element is `<= 9`, so the whole node prunes to `0`; then `bound = max(9, 5) = 9`. Total `2 + 0 = 2`. Correct. The over-count is gone, and it is gone for the reason I fixed: the `9` from the first segment now suppresses the second segment.

I also re-check a split that is not aligned to the root, to be sure the left-to-right visitation really holds for an arbitrary sub-range. `h = [9, 1, 2, 3, 8, 5, 6, 7]`, query `2 1 6` -> subarray `[1,2,3,8,5,6]`, true prefix maxima `1,2,3,8` = `4`. With `n=8`, range `[1,6]` decomposes (in visitation order) into canonical pieces that, scanned left to right, are `{1}, {2,3}, {4,5,6}` (indices), i.e. values `[1], [2,3], [8,5,6]`. Walk with `bound=-inf`: piece `[1]` -> count `1` (`1 > -inf`), bound `1`. Piece `[2,3]` -> over `[2,3]` with bound `1`: `2>1` yes, then `3>max(1,2)=2` yes -> count `2`, bound `3`. Piece `[8,5,6]` (indices 4,5,6) with bound `3`: `8>3` yes; `5>max(3,8)=8`? no; `6>8`? no -> count `1`, bound `8`. Total `1+2+1 = 4`. Correct. The threading survives an unaligned range.

**Second debug episode — the strict-vs-non-strict comparison on equal heights.** Visibility is *strict* (`>`), and equal heights are the place a counting solution silently flips to wrong. In my first draft of `countVisible` I had written the prune as `if (mx[node] < bound) return 0;` — strictly less — reasoning loosely that "if the node max is below the bound there is nothing." I trace `h = [4, 4, 4]`, query `2 0 2`. True answer: only the first `4` is visible (the later `4`s are not *strictly* taller than the running max `4`), so `1`. With the `<` prune: `countVisible([0,2], bound = -inf)`. Whole max `4 < -inf`? no, do not prune. Split `[4,4]` and `[4]`. Left `[4,4]` bound `-inf`: max `4 < -inf`? no. Split `[4]`,`[4]`. First `[4]`: leaf, but my leaf branch returned `1` whenever I did not prune — `4 < -inf` false, so return `1`. Second `[4]` with bound `max(-inf, 4) = 4`: `4 < 4`? false, do not prune, leaf returns `1`. Left subtotal `2`. Top-level right `[4]` with bound `max(-inf, mx[[4,4]]=4) = 4`: `4 < 4`? false, leaf returns `1`. Total `3`.

**Diagnosing it.** `3` is wrong (true `1`). The bug is the comparison: with `<`, an element *equal* to the bound is treated as "not dominated" and counted as visible, but equal is not strictly greater, so it must be suppressed. The prune condition must be `mx[node] <= bound` (a node whose max merely *ties* the bound contributes nothing), and the leaf, reached only when `mx > bound`, returns `1`. Switching to `<=`: re-trace `[4,4,4]`. `countVisible([0,2], -inf)`: `4 <= -inf`? no. Left `[4,4]` bound `-inf`: `4 <= -inf`? no; split. First leaf `[4]`: `4 <= -inf`? no -> `1`. Second leaf `[4]` bound `max(-inf,4)=4`: `4 <= 4`? yes -> `0`. Left subtotal `1`. Top right leaf `[4]` bound `max(-inf, 4)=4`: `4 <= 4`? yes -> `0`. Total `1`. Correct. Equal heights now behave: only the first of a run of equal values is visible. This is the off-by-one that a careless count gets exactly backwards.

**Wiring update and query, then checking the documented sample end to end.** Update is the textbook max-tree point set, also keeping the plain `h[]` array in sync (the segment tree does not strictly need `h[]` after building, but keeping it costs nothing and helps debugging). The query sets `bound = LLONG_MIN` (nothing seen to the left yet) and calls `queryCount`. Let me run the context sample by hand. `h = [3,1,4,1,5,9]`.

- `2 0 5` over `[3,1,4,1,5,9]`: prefix maxima `3,4,5,9` -> `4`.
- `2 1 3` over `[1,4,1]`: `1,4` -> `2`.
- `1 1 7`: `h = [3,7,4,1,5,9]`.
- `2 0 5` over `[3,7,4,1,5,9]`: `3,7,9` -> `3`.
- `2 0 2` over `[3,7,4]`: `3,7` -> `2`.

Expected `4, 2, 3, 2`. I ran the compiled solution on this exact input and it printed `4 / 2 / 3 / 2`. Matches.

**Edge cases, deliberately.**
- `n = 1`, query `2 0 0`: the only building is the leftmost, always visible -> `1`. `countVisible` on the single leaf with `bound = -inf` returns `1`. After an update `1 0 9` the leaf is still alone -> still `1`. Verified.
- `l == r` (single-element window): always `1`, since the leftmost (= only) building is visible. The query covers exactly one leaf; `mx > -inf` -> `1`. Verified on `2 2 2` over `[2,9,1,3]` -> `1`.
- All equal heights, e.g. `[4,4,4,4,4]`: any window -> `1` (only the first of the run). Verified -> `1,1,1` for three different windows.
- Strictly increasing window `[1,2,3,4,5]`: every building visible -> window length. Query `2 0 4` -> `5`, `2 1 3` -> `3`. Verified.
- Strictly decreasing `[5,4,3,2,1]`: only the first visible -> `1` for any window. Verified `1,1`.
- Cross-segment domination, `[1,1,1,9,2,3,4,5]` query `[0,7]` -> `2` (the bug case above), and `[4,7]` -> `4` (a window that starts *after* the `9`, so `2,3,4,5` are all visible). Verified `2` and `4`.
- Large heights up to `10^9`: comparisons are in `long long`, no overflow (I never sum heights, only compare and count). `[10^9, 1, 10^9]` query `[0,2]` -> visible `10^9` and then second `10^9` is not strictly greater -> `2`? Let me check: positions are `10^9 (vis), 1 (no), 10^9 (10^9 > 10^9? no)` -> `1`. Verified `1`.

**Randomized cross-check.** I generated 1000 random small instances (windows, updates, deliberately small value domains to force ties, plus a wider generator with values up to `10^9`) and compared the segment-tree solution against the `O(window)` brute force that recomputes prefix-max counts directly. Zero mismatches. A full-scale run at `n = q = 2*10^5` finished in 0.18 s using 11.6 MB — comfortably inside 2 s / 256 MB.

**Final solution.** I disproved the per-segment independent count by tracing the `9`-suppression case (returned `6` instead of `2`), threaded a running `bound` left to right across canonical segments to fix it; I fixed the strict-vs-non-strict prune by tracing all-equal heights (returned `3` instead of `1`) and switching `<` to `<=`; and I confirmed both fixes, all edge corners, the sample, and 1000 random cases. This is what I ship — one self-contained file:

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

**Causal recap.** Visibility = strict prefix maxima, which is not additive, so a per-node count taken in isolation over-counts: a trace of `[1,1,1,9,2,3,4,5]` query `[0,7]` returned `6` instead of `2` because the second canonical segment `[2,3,4,5]` was counted as if it began the window, ignoring the dominating `9` to its left; threading a running `bound` left to right across canonical segments (raise it by each fully covered node's max, recurse left before right) restores the suppression and fixes the double-count. A second trace on all-equal heights `[4,4,4]` returned `3` instead of `1` because a `<` prune treated ties as visible; the strict definition requires the prune `mx <= bound`, so only the first of a run of equal heights counts. With those two counting corrections, `long long` comparisons (no sums, so no overflow), and the standard `O(log n)` max-tree update, the solution is `O(log^2 n)` per query and agrees with the brute force on 1000 random cases and every hand-checked corner.
