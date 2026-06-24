**Problem.** Given `a[0..n-1]` (values may be negative), pick a subset with no two adjacent positions, maximizing the sum; the empty set is allowed, so the answer is at least `0`. Read `n` and the values from stdin, print the maximum sum.

**Why the obvious greedy is wrong.** "Repeatedly take the largest free positive value and block its neighbours" fails because the adjacency constraint is global. On `[8, 9, 2, 9, 9, -2, 8, -5]` greedy grabs `9 + 9 + 8 = 26`, but indices `0, 2, 4, 6` give `8 + 2 + 9 + 8 = 27`; blocking a neighbour can cost more than the value gained. Greedy is discarded.

**Key idea — linear prefix DP.** Scan left to right carrying two values: `skip` = best sum over the prefix with the last position **not** taken, and `take` = best with it taken. Transitions, both reading the *previous* pair:

- `take_i = skip_{i-1} + a[i]`  (taking `i` forces `i-1` skipped)
- `skip_i = max(skip_{i-1}, take_{i-1})`  (skipping `i` leaves `i-1` free)

Initialize `skip = 0` (empty prefix) and `take = -inf` (no last-taken state before any element). Answer: `max(take, skip, 0)`.

**Two pitfalls to get right.**
1. *In-place update.* Compute both new values from the old `(skip, take)` via temporaries. Updating `skip` first and then using it for `take` builds `take` on a state that already took `i-1` — illegal adjacency. (A trace of `[1, 1]` returning `2` exposes exactly this.)
2. *Overflow.* With `n` up to `2*10^5` and `|a[i]|` up to `10^9`, sums reach `~2*10^14`; use `long long`. An `int` is a silent wrong-answer on large tests.

**Edge cases (all handled by the recurrence + the final `max(...,0)`):** `n = 0` -> `0`; a single negative -> `0`; all negatives -> `0`.

**Complexity.** `O(n)` time, `O(1)` extra space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> answer 0
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // skip = best prefix sum with the last position NOT taken (starts 0: empty set);
    // take = best with the last position taken (starts -inf: impossible before any element).
    long long take = LLONG_MIN / 4, skip = 0;
    for (int i = 0; i < n; i++) {
        long long ntake = skip + a[i];     // take i  => i-1 skipped
        long long nskip = max(skip, take); // skip i  => i-1 either
        take = ntake;
        skip = nskip;
    }

    cout << max({take, skip, 0LL}) << "\n"; // empty selection always allowed
    return 0;
}
```
