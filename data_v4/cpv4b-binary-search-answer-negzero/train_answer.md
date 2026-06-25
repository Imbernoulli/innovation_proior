**Problem.** Given `n` borehole coordinates `x[0..n-1]` (integers that may be negative, zero, or
duplicated) and a count `k` (`2 <= k <= n`), choose `k` of them so that the minimum distance between
any two chosen coordinates is as large as possible; print that maximum *isolation*. Read `n k` then the
`n` coordinates from stdin. The answer is always a non-negative integer.

**Key idea — binary search on the answer.** After sorting the coordinates, the isolation of a choice is
just the minimum gap between *adjacent* chosen coordinates. The predicate "can `k` sensors be placed
pairwise at least `d` apart?" is monotone in `d`, so binary-search the largest feasible `d` over
`[0, span]` where `span = x[n-1] - x[0]`. The feasibility test is a linear greedy on the sorted array:
keep the leftmost borehole, then repeatedly take the next borehole whose coordinate is at least
`last_placed + d`; the placement is feasible iff at least `k` sensors get placed. Placing each sensor as
far left as legal is optimal (any valid placement shifts left to the greedy without losing count), so
`O(n log span)` total.

**Pitfalls.**
1. *Wrong base case / lower bound.* The smallest legal isolation is `0` (duplicate coordinates, or
   `k = n` over a non-strictly-increasing array), so the search interval must start at `lo = 0`, not
   `lo = 1`. With `lo = 1` the search can never certify the answer `0` and is correct only by the
   accident of the accumulator being initialized to `0`; pick a sentinel like `-1` and it returns `-1`.
   Setting `lo = 0` makes `feasible(0)` — always true, since sorted coordinates satisfy `x[i] - last >= 0`
   — establish the answer by construction.
2. *Comparison direction.* Use `x[i] - last >= d` (not `> d`). At `d = 0` the `>=` lets equal coordinates
   count so `feasible(0)` reaches `k`; with `>` duplicates are rejected and a genuinely-zero answer
   becomes "infeasible," giving garbage.
3. *Sign of the bound.* `span = x[n-1] - x[0]` is a difference of sorted endpoints, hence `>= 0` even when
   every coordinate is negative — never bound the search with a raw coordinate.
4. *Overflow.* A span can reach `2*10^9` (coordinates near `+-10^9`); keep coordinates, the span, and the
   midpoint in `long long`, and form the midpoint as `lo + (hi - lo) / 2`.

**Edge cases (all handled by `lo = 0`, the `>=`, and the always-true `feasible(0)`):** all-identical
coordinates -> `0`; all-negative coordinates -> a positive distance from their spread; `k = n` (forced)
-> minimum adjacent gap, possibly `0`; `k = 2` -> `x[n-1] - x[0]`.

**Complexity.** Sorting `O(n log n)`; binary search `O(log span)` iterations, each an `O(n)` greedy, so
`O(n (log n + log span))` time and `O(n)` space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, k;
    if (!(cin >> n >> k)) return 0;
    vector<long long> x(n);
    for (auto &v : x) cin >> v;

    sort(x.begin(), x.end());

    // feasible(d): can we install k sensors so every pair is at least d apart?
    // Greedy on the sorted coordinates: always keep the first borehole, then take the
    // next borehole whose coordinate is at least (last placed + d).
    auto feasible = [&](long long d) -> bool {
        int cnt = 1;                 // place the first (leftmost) sensor
        long long last = x[0];
        for (int i = 1; i < n; i++) {
            if (x[i] - last >= d) {
                cnt++;
                last = x[i];
            }
        }
        return cnt >= k;
    };

    // The optimal isolation is some adjacent-difference value, in [0, span].
    long long lo = 0, hi = x[n - 1] - x[0];

    long long ans = 0;
    while (lo <= hi) {
        long long mid = lo + (hi - lo) / 2;
        if (feasible(mid)) { ans = mid; lo = mid + 1; }
        else hi = mid - 1;
    }

    cout << ans << "\n";
    return 0;
}
```
