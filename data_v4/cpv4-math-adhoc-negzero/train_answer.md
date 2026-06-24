**Problem.** Given `a[0..n-1]` (`-2 <= a[i] <= 2`, `0 <= n <= 62`), consider every contiguous subarray and the product of its elements. The empty subarray is allowed and its product is the empty product `1`. Output the largest product over all subarrays; because the empty subarray is always available, the answer is at least `1`. Read `n` and the values from stdin, print the maximum product.

**Why the single-running-maximum (Kadane) idea is wrong.** Carrying only the best product ending at each index fails on products because a negative element flips signs. On `[2, -2, -2]` a single running max keeps `-2` over `-4` at index 1 (it is the larger), then index 2 gives `(-2) * (-2) = 4` — but the discarded `-4` would have become `(-4) * (-2) = 8`, the true answer. The most-negative running product is exactly what a future negative needs, and a single maximum throws it away.

**Key idea — track running max AND min.** For a non-empty subarray ending at `i`, keep both the largest product `curMax` and the smallest product `curMin`. A non-empty window ending at `i` is the singleton `a[i]` or an extension of a non-empty window ending at `i-1`, so the candidates are `a[i]`, `curMax_{i-1} * a[i]`, `curMin_{i-1} * a[i]`:

- `curMax_i = max(a[i], curMax_{i-1} * a[i], curMin_{i-1} * a[i])`
- `curMin_i = min(a[i], curMax_{i-1} * a[i], curMin_{i-1} * a[i])`

Including `a[i]` itself restarts the window (crucial across a zero). The answer is `max(1, curMax over all i)`, where the `1` is the empty subarray.

**Pitfalls.**
1. *Base case / sign floor.* Seed the answer at `1` (empty subarray) and start the first window at `a[0]` itself — do not seed `curMax`/`curMin` at a phantom empty run of product `1`. That phantom only "works" because `max({a[i], ...})` re-includes the singleton and washes the seed out at index 0; it violates the invariant "`curMax`/`curMin` are non-empty runs" and breaks under any natural refactor. Track a `haveRun` flag and extend only when a non-empty run exists.
2. *Sign handling.* You must carry the minimum, not just the maximum, because a negative `a[i]` turns the most-negative product into the most-positive one.
3. *Empty / all-negative / all-zero.* `max(best, 1)` returns `1` exactly when no non-empty window beats the empty subarray: `n = 0`, a lone negative, all-negative of odd length where one factor must be dropped, and all-zeros.
4. *Overflow.* With `n <= 62` and `|a[i]| <= 2`, products reach `2^62`; use `long long` (fits in signed 64-bit), an `int` overflows.

**Edge cases (all handled by the recurrence + the `haveRun` base case + the `1` floor):** `n = 0` -> `1`; a lone negative or lone zero -> `1`; all negatives of odd length -> drop one factor; all zeros -> `1`; interior zeros reset the run so no product spans them.

**Complexity.** `O(n)` time, `O(1)` extra space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> empty subarray, product 1
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // The empty subarray is allowed and its product is defined as 1, so the answer
    // is at least 1. For a NON-EMPTY subarray ending exactly at position i we track
    // BOTH the largest product (curMax) and the smallest product (curMin), because a
    // negative a[i] turns the most-negative running product into the most-positive one.
    long long best = 1;                    // empty subarray: product 1, always available
    long long curMax = 0, curMin = 0;      // products of best/worst NON-EMPTY run ending at i-1
    bool haveRun = false;                  // does such a non-empty run ending at i-1 exist yet?

    for (int i = 0; i < n; i++) {
        long long x = a[i];
        long long nMax, nMin;
        if (!haveRun) {
            // nothing to extend: the only non-empty window ending at i is {x}
            nMax = x;
            nMin = x;
        } else {
            long long e1 = curMax * x;     // extend the best run by x
            long long e2 = curMin * x;     // extend the worst run by x (matters when x < 0)
            nMax = max({x, e1, e2});       // start fresh at x, or take the better extension
            nMin = min({x, e1, e2});
        }
        curMax = nMax;
        curMin = nMin;
        haveRun = true;
        best = max(best, curMax);          // compare against empty (1) and earlier windows
    }

    cout << best << "\n";
    return 0;
}
```
