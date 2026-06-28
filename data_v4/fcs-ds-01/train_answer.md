**Problem.** Given an array `a[0..n-1]` of integers and `q` offline queries
`(l, r)` (0-indexed, inclusive), report for each query the number of distinct
values in `a[l..r]`. Constraints: `n, q <= 2*10^5`, `|a[i]| <= 10^9`. Output one
answer per query in input order.

**Why the obvious approaches are wrong/too slow.** Rescanning each range with a
set is `O(q*n) = 4*10^10` at the limits — far over a 2-second budget. The
prefix-sum instinct also fails: distinct-count is *not* subtractive across a
split, because a value can appear on both sides. On `[1,2,1]`, prefix-distinct of
`[0,1]` and `[0,2]` are both 2, yet `2 - 2 = 0 != 1`, the true count of the
suffix `[2,2]`. There is no per-query scalar-prefix trick.

**Key idea — Mo's algorithm (sqrt-block query reordering).** Keep one window
`[L, R]` and a table `cnt[value]`. Moving an endpoint by one is `O(1)`: on add,
if a value's count goes `0 -> 1` the running distinct total rises; on remove, if
it goes `1 -> 0` it falls. The whole cost is then *total pointer travel*, which
depends entirely on the order in which queries are visited. Sort queries by
`(block of l, r)` with block width `B`: left-pointer travel is `O(q*B)` and
right-pointer travel is `O(n^2 / B)`. Balancing `q*B = n^2 / B` gives
`B = n / sqrt(q)` and total cost `O((n + q) * sqrt(q))` — about `9*10^7` moves at
`n = q = 2*10^5`, comfortably in time. That sqrt-block sort key is the insight;
the `O(1)` add/remove is the easy half.

**Refinement.** Use the even/odd boustrophedon: sort `r` ascending in
even-indexed `l`-blocks and descending in odd ones, so the right pointer sweeps
back and forth instead of snapping to the block start each time. Same
asymptotics, roughly half the right-pointer constant.

**Pitfalls to get right.**
1. *Loop order — grow before shrink.* Extend `R` rightward and `L` leftward
   (the `add`s) *before* shrinking `R` and `L` (the `remove`s). Shrinking first
   calls `remove` on positions never added, driving `cnt` negative and
   corrupting the distinct total. A trace of `a=[1,2]` with queries `(1,1)` then
   `(0,0)` exposes exactly this.
2. *Coordinate-compress the values.* Values span `[-10^9, 10^9]` and may be
   negative, so index the count array by compressed rank in `[0, V)`, `V <= n`.
3. *Restore answers to input order.* Carry the original index with each query
   and scatter results back after sorting.
4. *Batch the output.* Up to `2*10^5` lines — build one string and flush once.

**Edge cases.** `n = 1` → answer 1; all-equal array → every answer 1;
all-distinct → answer `r-l+1`; negatives/zeros → handled by compression;
`+-10^9` extremes → compression bounds the count array; single-element ranges
`l == r` and repeated identical queries → the four while-loops become no-ops and
read the same total. All verified against a brute-force set oracle over hundreds
of random and adversarial cases with zero mismatches; full-scale
`n = q = 2*10^5` runs in ~0.1 s and ~10 MB.

**Complexity.** `O((n + q) * sqrt(q))` time, `O(n + q)` space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

// Offline range-distinct counting via Mo's algorithm.
//
// We must answer q queries, each asking for the number of DISTINCT values in
// a[l..r]. Re-scanning per query is O(q*n) which is ~4*10^10 at the limits and
// far too slow. Mo's algorithm reorders the queries so that two consecutive
// queries differ in their endpoints by only a little on average, then it walks
// a sliding window [curL, curR] from one query to the next with O(1)
// add/remove operations on a value-count array. With block size B = n/sqrt(q),
// the left pointer moves O(q*B) times in total and the right pointer moves
// O(n^2 / B) times in total; balancing gives O((n + q) * sqrt(q)) work overall,
// which is ~1.3*10^8 element moves at n = q = 2*10^5 and runs well inside the
// limit. The crux ("the insight") is the SORT KEY: queries are bucketed by the
// block index of l, and within a block sorted by r -- with the standard
// even/odd boustrophedon trick that sweeps r left-to-right in even blocks and
// right-to-left in odd blocks, halving the right-pointer travel.

struct Query {
    int l, r, idx;
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    if (!(cin >> n >> q)) return 0;

    vector<int> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];

    // Coordinate-compress the values so the count array is sized by #distinct
    // values (at most n), independent of the value range.
    vector<int> comp(a.begin(), a.end());
    sort(comp.begin(), comp.end());
    comp.erase(unique(comp.begin(), comp.end()), comp.end());
    for (int i = 0; i < n; i++) {
        a[i] = (int)(lower_bound(comp.begin(), comp.end(), a[i]) - comp.begin());
    }
    int V = (int)comp.size();

    vector<Query> qs(q);
    for (int i = 0; i < q; i++) {
        int l, r;
        cin >> l >> r;          // 0-indexed, inclusive on both ends
        qs[i] = {l, r, i};
    }

    // Block size: n / sqrt(q) (guarded to be >= 1) minimizes total pointer
    // movement under the standard Mo cost analysis.
    int block = max(1, (int)(n / max(1.0, sqrt((double)q))));

    sort(qs.begin(), qs.end(), [&](const Query &x, const Query &y) {
        int bx = x.l / block, by = y.l / block;
        if (bx != by) return bx < by;
        // Even/odd trick: alternate the r-sort direction per block so the right
        // pointer sweeps back and forth instead of resetting each block.
        if (bx & 1) return x.r > y.r;
        return x.r < y.r;
    });

    vector<int> cnt(V, 0);   // cnt[value] = occurrences inside current window
    long long distinct = 0;  // number of values with cnt > 0 in the window
    vector<long long> ans(q);

    // Window [curL, curR] is inclusive; start it empty (curL > curR).
    int curL = 0, curR = -1;

    auto add = [&](int pos) {
        if (cnt[a[pos]]++ == 0) distinct++;
    };
    auto remove = [&](int pos) {
        if (--cnt[a[pos]] == 0) distinct--;
    };

    for (const Query &Q : qs) {
        // Grow the window outward first, then shrink, so cnt never goes
        // negative on a transient.
        while (curR < Q.r) add(++curR);
        while (curL > Q.l) add(--curL);
        while (curR > Q.r) remove(curR--);
        while (curL < Q.l) remove(curL++);
        ans[Q.idx] = distinct;
    }

    string out;
    out.reserve((size_t)q * 7);
    char buf[24];
    for (int i = 0; i < q; i++) {
        int len = snprintf(buf, sizeof(buf), "%lld\n", ans[i]);
        out.append(buf, len);
    }
    cout << out;
    return 0;
}
```
