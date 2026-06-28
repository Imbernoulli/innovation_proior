**Problem.** Given `a[0..n-1]` (values may be negative), look at every contiguous non-empty subarray and allow deleting at most one element from it, with the remainder required to stay non-empty (a length-`1` subarray keeps its element, it cannot delete it). Maximize the resulting sum. Read `n` and the values from stdin, print the maximum score. Because a length-`1` subarray with no deletion is always legal, the answer is defined even for all-negative input (then it is the largest single element).

**Why the obvious greedy is wrong.** The reflex is "run plain Kadane to find the best subarray, then delete the most negative element inside that window." It fails because plain Kadane picks its window under the *no-deletion* rule, so the window that is best to delete from can be different — and larger. On `a = [5, 6, -3, 3, -5]`, plain Kadane's best window is `[5, 6]` with sum `11` (the `-3` is a reason it stops there), and its most negative element is the positive `5`, so deleting helps nothing and greedy reports `11`. But the true optimum takes the window `[5, 6, -3, 3]` and deletes the `-3`, leaving `5 + 6 + 3 = 14`. The deleted element is exactly the one that made plain Kadane stop early, so the greedy never considers the right window. Greedy is discarded.

**Key idea — two-state Kadane DP.** Scan left to right carrying two values, both "best subarray *ending at* `i`":

- `noDel` = best sum with **no** element deleted,
- `oneDel` = best sum with **exactly one** element deleted (remainder still non-empty).

Transitions, all reading the *previous* `(noDel, oneDel)`:

- `noDel_i = max(a[i], noDel_{i-1} + a[i])`  (ordinary Kadane: start fresh or extend)
- `oneDel_i = max(noDel_{i-1}, oneDel_{i-1} + a[i])`  where `noDel_{i-1}` = "append `a[i]` to a no-deletion window ending at `i-1`, then delete `a[i]`", and `oneDel_{i-1} + a[i]` = "deletion already spent earlier, keep `a[i]`"

Seed `noDel = oneDel = -inf` (no subarray exists before index 0). At `i = 0` this makes `noDel_0 = a[0]` and `oneDel_0 = -inf`, correctly forbidding deletion of a lone element. The answer is the max of `noDel_i` and `oneDel_i` over all `i`.

**Two pitfalls to get right.**
1. *In-place update order.* `oneDel_i` reads `noDel_{i-1}`. Compute both new values from the old pair via temporaries and only then assign. Updating `noDel` first and reading it for `oneDel` lets `oneDel` "delete `a[i]`" from a window that already started fresh at `a[i]` — illegal. (A trace of `[10, -100, 10]` returning `10` instead of `20` exposes exactly this.)
2. *Overflow.* With `n` up to `10^5` and `|a[i]|` up to `10^9`, sums reach `~10^14`; use `long long`. An `int` is a silent wrong-answer on large tests. The negative sentinel `LLONG_MIN/4` has room to absorb `+a[i]` without underflowing.

**Edge cases (handled by the recurrence + the seed):** `n = 1` -> `a[0]` (no deletion to empty); all-negative -> the largest single element; the deletion never produces an empty remainder because the "delete `a[i]`" branch only beats `-inf` once a real `noDel_{i-1}` exists (i.e. from `i >= 1`).

**Complexity.** `O(n)` time, `O(1)` extra space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // The subarray must be non-empty AFTER any deletion, so we need n >= 1.
    // noDel = best sum of a subarray ending at i with NO element deleted.
    // oneDel = best sum of a subarray ending at i with EXACTLY one element deleted.
    const long long NEG = LLONG_MIN / 4;
    long long noDel = NEG, oneDel = NEG;
    long long best = NEG;
    for (int i = 0; i < n; i++) {
        // oneDel must be computed from the PREVIOUS noDel/oneDel before noDel is updated.
        long long newOneDel = max(noDel,                 // delete a[i]; segment so far = old noDel ending at i-1
                                  oneDel + a[i]);        // deletion already used earlier; extend by a[i]
        long long newNoDel = max(a[i], noDel + a[i]);    // standard Kadane: start fresh or extend
        noDel = newNoDel;
        oneDel = newOneDel;
        best = max(best, noDel);
        best = max(best, oneDel);
    }

    cout << best << "\n";
    return 0;
}
```
