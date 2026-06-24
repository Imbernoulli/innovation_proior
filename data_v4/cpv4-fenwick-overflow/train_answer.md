**Problem.** Given `n` positive integers `a[0..n-1]`, sum `a[i] * a[j]` over every pair `i < j` with `a[i] > a[j]` (an inversion). Equal values do not count. Read `n` and the values from stdin; print the weighted inversion sum. A non-decreasing sequence gives `0`.

**Why the naive loop is wrong (too slow).** The double loop over pairs is `O(n^2) ~ 5*10^9` at `n = 10^5` and blows the 1-second limit. We need `O(n log n)`.

**Key idea — Fenwick sweep, decompose by larger index.** Every inversion has a unique larger index `j`. The inversions ending at `j` are the earlier positions `i < j` with `a[i] > a[j]`, each contributing `a[i] * a[j]`, so

- `contribution(j) = a[j] * ( sum of a[i] over i < j with a[i] > a[j] )`,

and the answer is the sum of `contribution(j)`. Sweep `j` left to right keeping, in a Fenwick tree indexed by compressed value-rank, the **sum of values** of already-inserted (earlier) elements. The sum of earlier values strictly greater than `a[j]` is `query(m) - query(r)`, where `r` is `a[j]`'s rank and `m` the number of distinct values. Multiply by `a[j]`, add to the answer, then insert `a[j]`. Query *before* insert so an element never pairs with itself or a later element.

**Pitfalls.**
1. *Silent int overflow on the value path (the trap).* A single product is up to `30000 * 29999 ~ 9*10^8`, and the total reaches `~4.5*10^18`. Crucially, the Fenwick tree's *stored value-sums* and the `query(m) - query(r)` difference also exceed `2^31` on big-value inputs. If the tree storage or the query return is `int`, it wraps **before** the multiply, and casting the final product to `long long` cannot recover it. Make the tree storage, the query return, and the answer all `long long`. (On `[30000,30000,30000,29999,29999,29999]` the true answer is `8099730000`; an all-`int` version prints the tell-tale negative `-490204592`.)
2. *Strict vs. non-strict range.* Ties are not inversions. Use `query(m) - query(r)` (excludes rank `r` itself); `query(m) - query(r-1)` would wrongly count equal earlier elements.
3. *Insert order.* Query first, then insert, so the `i < j` constraint holds.

**Edge cases.** `n = 0` -> `0` (loop never runs); `n = 1` -> `0` (no pairs); strictly increasing -> `0` (no earlier element is greater); all equal -> `0` (ties excluded). All fall out of the recurrence with the `long long` accumulator.

**Complexity.** `O(n log n)` time (one sort for compression plus `n` Fenwick query/update pairs), `O(n)` space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> answer 0
    vector<int> a(n);
    for (auto &x : a) cin >> x;

    // Coordinate-compress the values so the Fenwick index space is O(n).
    vector<int> srt(a);
    sort(srt.begin(), srt.end());
    srt.erase(unique(srt.begin(), srt.end()), srt.end());
    int m = (int)srt.size();
    auto rankOf = [&](int v) {
        return int(lower_bound(srt.begin(), srt.end(), v) - srt.begin()) + 1; // 1..m
    };

    // Two Fenwick trees indexed by compressed value:
    //   cnt[r] = how many earlier elements have value-rank r,
    //   sum[r] = sum of those earlier element VALUES (this is the part that overflows int).
    vector<long long> bitCnt(m + 1, 0), bitSum(m + 1, 0);
    auto add = [&](vector<long long> &bit, int i, long long delta) {
        for (; i <= m; i += i & (-i)) bit[i] += delta;
    };
    auto query = [&](vector<long long> &bit, int i) {
        long long s = 0;
        for (; i > 0; i -= i & (-i)) s += bit[i];
        return s;
    };

    long long answer = 0;
    // Sweep left to right. For element j (value v, rank r), every earlier element with a
    // strictly greater value forms an inversion (i<j, a[i] > a[j]); each contributes a[i]*a[j].
    // Summed over those earlier elements that is v * (sum of their values).
    for (int j = 0; j < n; j++) {
        long long v = a[j];
        int r = rankOf(a[j]);
        // earlier elements with rank in (r, m]  =>  value strictly greater than a[j].
        long long greaterValueSum = query(bitSum, m) - query(bitSum, r);
        answer += v * greaterValueSum;     // v fits int but the product / accumulator do not
        // insert this element
        add(bitCnt, r, 1);
        add(bitSum, r, v);
    }

    cout << answer << "\n";
    return 0;
}
```
