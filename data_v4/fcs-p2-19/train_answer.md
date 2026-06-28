**Problem.** Given an array `a[0..n-1]`, find the length of the longest *bitonic* subsequence: elements taken in their original order whose values strictly increase to a single peak and then strictly decrease. At least one real increase is required (the array must go up before it comes down), the strictly-decreasing tail may be empty (a strictly increasing run of length `>= 2` counts), and both comparisons are strict (equal values cannot be adjacent in the chosen subsequence). Read `n` and the values from stdin; print the length, or `0` if no valid bitonic subsequence exists (`n <= 1`, a strictly decreasing array, or a plateau of equal values).

**Why the obvious single pass is wrong.** "Walk the array once, climbing while the next element rises and then descending while it falls, tracking the longest mountain" solves the longest *contiguous substring* mountain, not the longest *subsequence* mountain. A subsequence may delete elements on either slope, which a neighbour-based walk cannot express. On `[1, 11, 2, 10, 4, 5, 2, 1]` the single pass scrapes together short contiguous mountains of length about `4`, but the true answer is `6`: the subsequence `1 < 2 < 10 > 4 > 2 > 1` skips the `11` at index 1 entirely and stitches indices `0, 2, 3` across the gap. The walk is discarded.

**Key idea — longest-increasing-from-the-left glued to longest-decreasing-from-the-right at each peak.** At `n <= 5000` an `O(n^2)` method is comfortably in budget (about `5 * 10^7` comparisons, ~0.03 s), so I pick the method I can prove. Every bitonic subsequence has exactly one peak; to its left it is a strictly increasing subsequence ending at the peak, to its right a strictly decreasing subsequence starting at the peak, and these halves are independent given the peak. So for each index `i` compute:

- `inc[i]` = length of the longest strictly increasing subsequence **ending** at `i`, via `inc[i] = 1 + max{ inc[j] : j < i, a[j] < a[i] }` (default `1`).
- `dec[i]` = length of the longest strictly decreasing subsequence **starting** at `i`, via `dec[i] = 1 + max{ dec[j] : j > i, a[j] < a[i] }` (default `1`).

The best bitonic subsequence peaking at `i` has length `inc[i] + dec[i] - 1` (the `-1` counts the shared peak once). The answer is the maximum of this over all valid peaks.

**Two corners to get right.**
1. *Must actually increase.* Gate every peak on `inc[i] >= 2`. Without it, a strictly decreasing array `[3, 2, 1]` has `inc = [1,1,1]`, `dec = [3,2,1]`, and the guardless combine reports `1 + 3 - 1 = 3` — treating a pure descent as bitonic. The `inc[i] >= 2` guard encodes the `p >= 1` "real ascent" rule, so strictly decreasing arrays and equal-value plateaus correctly return `0`. (A trace of `[3, 2, 1]` returning `3` exposes exactly this.)
2. *Strictness.* Both DPs use the strict predicate `a[j] < a[i]`, so equal values never extend a chain — `[3,3,3]` yields `0`, and the empty descent (`dec[i] = 1`) makes a full strictly increasing array return its length `n`.

**Edge cases (all handled by the recurrence + the `inc[i] >= 2` gate):** `n = 0` and `n = 1` -> `0`; strictly decreasing -> `0`; all-equal plateau -> `0`; strictly increasing `[1,2,3,4]` -> `4` (full ascent, empty descent).

**Complexity.** `O(n^2)` time, `O(n)` extra space. The answer is a length bounded by `n <= 5000`, so an `int` is safe everywhere — no overflow concerns.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> answer 0
    vector<long long> a(n);
    for (auto &x : a) cin >> x;
    if (n == 0) { cout << 0 << "\n"; return 0; }

    // inc[i] = length of the longest STRICTLY increasing subsequence that ENDS at i.
    // dec[i] = length of the longest STRICTLY decreasing subsequence that STARTS at i.
    // A bitonic subsequence with peak at index i (the peak counted once) has length
    // inc[i] + dec[i] - 1. We require an actual increase before the peak, i.e. inc[i] >= 2,
    // so a purely non-increasing array yields answer 0 (no valid increase-then-decrease).
    vector<int> inc(n, 1), dec(n, 1);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < i; j++)
            if (a[j] < a[i])
                inc[i] = max(inc[i], inc[j] + 1);

    for (int i = n - 1; i >= 0; i--)
        for (int j = i + 1; j < n; j++)
            if (a[j] < a[i])
                dec[i] = max(dec[i], dec[j] + 1);

    int best = 0;
    for (int i = 0; i < n; i++)
        if (inc[i] >= 2)                          // peak must be preceded by a real increase
            best = max(best, inc[i] + dec[i] - 1);

    cout << best << "\n";
    return 0;
}
```
