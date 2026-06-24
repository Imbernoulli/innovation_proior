**Problem.** Given `n` integers `a[0..n-1]` (values may be negative, zero, or positive) and a threshold `T`, count the unordered pairs of distinct positions `(i, j)` with `i < j` and `a[i] + a[j] <= T`. Read `n`, `T`, and the values from stdin; print the count.

**Key idea — sort + two-pointer sweep.** Sort `a` ascending and sweep `lo` up from the front and `hi` down from the back.

- If `a[lo] + a[hi] <= T`: since the array is sorted, `a[lo]` paired with each of `a[lo+1..hi]` also satisfies the bound (each such partner is `<= a[hi]`), so `a[lo]` contributes `hi - lo` valid pairs. Add `hi - lo` and advance `lo`.
- If `a[lo] + a[hi] > T`: then `a[hi]` paired with the smallest remaining low already exceeds `T`, and every other remaining low is `>= a[lo]`, so `a[hi]` pairs with nothing in `[lo, hi-1]`. Drop `hi` by one.

Each pair is counted exactly once (always anchored on the current `lo` when it fits). Loop while `lo < hi`. Answer accumulates in the loop. Complexity is dominated by the sort.

**Pitfalls.**
1. *Loop guard.* Use `while (lo < hi)`, not `lo <= hi`. At `lo == hi` the two pointers reference the *same* element, so `a[lo] + a[hi]` would pair an element with itself — not a valid `(i, j)` with `i < j`. (A trace of `[-4, -4]` with `T = -8` returning `2` instead of `1` exposes exactly this.)
2. *Count per step.* Add `hi - lo`, not `1`. When the sum fits, `a[lo]` pairs with *all* `hi - lo` partners above it; `ans += 1` undercounts whenever more than one partner exists.
3. *Overflow, twice.* With `|a[i]|` up to `10^9`, a pair sum reaches `2*10^9`, and with `n` up to `2*10^5` the count reaches `~2*10^10`. Both exceed 32-bit range, so the values (so the sum is computed in 64-bit before comparing to `T`) **and** the accumulator must be `long long`. An `int` anywhere is a silent wrong-answer.
4. *Sign of the bound.* Use `<=` so a pair whose sum equals `T` counts; a flipped `<`, `>=`, or `>` miscounts, and the all-negative-with-negative-`T` tests are built to catch it.

**Edge cases (all handled by the sort + `lo < hi` sweep, no special casing):** `n = 0` -> `hi = -1`, loop never runs -> `0` (this is the base-case corner: `lo < hi` is safe even with `hi = -1`); `n = 1` -> no pair -> `0`; all-negative with negative `T` (e.g. `[-3,-3,-3]`, `T = -7`) -> every sum `-6 > -7` -> `0`; all-zero with `T = 0` -> every pair counts; `sum == T` boundary -> counted via `<=`.

**Complexity.** `O(n log n)` time (the sort; the sweep is `O(n)`), `O(1)` extra space beyond the input.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long T;
    if (!(cin >> n >> T)) return 0;        // empty input -> nothing to do
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    sort(a.begin(), a.end());

    // Count unordered pairs (i<j) with a[i]+a[j] <= T via a two-pointer sweep.
    // After sorting, for the pair anchored at the high end `hi`, every index in
    // [lo, hi-1] forms a valid pair when a[lo]+a[hi] <= T (monotone in lo).
    long long ans = 0;
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        if (a[lo] + a[hi] <= T) {
            ans += (long long)(hi - lo);   // all of lo..hi-1 pair with hi
            lo++;
        } else {
            hi--;
        }
    }

    cout << ans << "\n";
    return 0;
}
```
