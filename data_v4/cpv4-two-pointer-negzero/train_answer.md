**Problem.** Given `n` thruster impulses `a[0..n-1]` (values may be negative or zero) and a threshold `T` (also possibly negative or zero), count the unordered pairs `{i, j}`, `i != j`, with `a[i] + a[j] >= T`. Read `n`, then `T`, then the values from stdin; print the count. With `n = 0` or `n = 1` no pair exists, so the answer is `0`.

**Why brute force is too slow.** Enumerating every pair is `O(n^2)`, about `2*10^10` operations at `n = 2*10^5` — far over the time limit. Sorting unlocks a faster count.

**Key idea — sort + converging two pointers.** Sort `a` ascending and run `lo = 0`, `hi = n-1`. While `lo < hi`:

- If `a[lo] + a[hi] >= T`, then for every `k` in `[lo, hi-1]` we have `a[k] >= a[lo]`, so `a[k] + a[hi] >= T` too — that is `hi - lo` qualifying pairs at once. Add `hi - lo`, then `hi--`.
- Otherwise `a[lo]` is too small to reach `T` even with the largest remaining `hi`, so it qualifies with nothing left: `lo++`.

Pairing `hi` only with strictly-smaller positions counts each unordered pair exactly once. `O(n log n)` for the sort, `O(n)` for the sweep.

**Correctness.** The invariant is that all admissible pairs among indices still in `[lo, hi]` are either counted now or remain reachable. In the `>=` branch, `hi` is the largest available value, so its admissible partners are exactly the contiguous block `[lo, hi-1]` (ascending order makes "qualifies" upward-closed); counting them and retiring `hi` loses nothing. In the `<` branch, `a[lo]` plus the largest remaining value already falls short, so `a[lo]` is admissible with nothing left and can be dropped. Each step removes one element, so the sweep terminates in `O(n)`.

**Pitfalls.**
1. *Off-by-one in the increment.* The count for the current `hi` is `hi - lo` (the closed range `[lo, hi-1]`, which **includes** `lo`), not `hi - lo - 1`. The wrong form drops the pair `{lo, hi}` itself: on `n=2, T=1, a=[-2,8]` it returns `0` instead of `1`.
2. *Sign / overflow.* Pair sums range in `[-2*10^9, 2*10^9]` and `|T|` can be `2*10^9`, both beyond 32-bit; the count can reach `n*(n-1)/2 ~ 2*10^10`. Read values and `T` as `long long`, form the sum in 64-bit, and keep the count in `long long`. A 32-bit comparison wraps near the bounds and flips admissibility (e.g. an all-negative pair `-1e9 + -1e9` against `T = -2e9`).
3. *Base case.* With `n < 2` there is no pair. Guard it explicitly and print `0`; relying on `hi = n-1 = -1` making the loop condition false is correctness by accident, not intent.

**Edge cases.** `n = 0` and `n = 1` -> `0` (guard). All-negative with an unreachable `T` -> `0`. All-negative with very low `T` -> `n*(n-1)/2`. All-zero at `T = 0` -> every pair counts (`>=`, not `>`). `T` above every reachable sum -> `0`.

**Complexity.** `O(n log n)` time (dominated by the sort), `O(1)` extra space beyond the input.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;                 // no input / n = 0 -> 0 pairs
    long long T;
    cin >> T;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    if (n < 2) {                               // fewer than two thrusters: no pair exists
        cout << 0 << "\n";
        return 0;
    }

    sort(a.begin(), a.end());

    // Count unordered pairs {i, j}, i < j (in sorted order), with a[i] + a[j] >= T.
    // Two converging pointers on the sorted array.
    long long count = 0;
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        if (a[lo] + a[hi] >= T) {
            // a[hi] paired with every index in [lo, hi-1] also satisfies the threshold,
            // because a is sorted ascending: those are (hi - lo) valid pairs.
            count += (long long)(hi - lo);
            hi--;
        } else {
            lo++;                              // smallest element can never reach T with this hi
        }
    }

    cout << count << "\n";
    return 0;
}
```
