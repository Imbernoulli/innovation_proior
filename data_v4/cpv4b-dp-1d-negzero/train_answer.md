**Problem.** A drone flies a corridor of `n` stops with signed charges `a[0..n-1]`. Choose one
contiguous **non-empty** block of stops; inside it you may re-route around **at most one** stop
(removing its contribution, provided at least one stop is still flown). Maximize the net charge.
Read `n` and the values from stdin, print the maximum. Note `1 <= n`, so the empty mission is
forbidden — when every charge is negative the answer is the **least-negative single stop**, a
negative number, not `0`.

**Key idea — linear two-state DP.** Scan left to right, carrying for the block *ending at the current
stop*:

- `f0` = best net charge with **no** skip used yet,
- `f1` = best net charge with the skip **already** used.

Transitions, both reading the *previous* pair:

- `f0_i = max(a[i], f0_{i-1} + a[i])`  — restart the block at `i`, or extend; never `max(0, …)`, since an empty block is illegal.
- `f1_i = max(f0_{i-1}, f1_{i-1} + a[i])`  — either skip stop `i` now (extend the no-skip block ending at `i-1`, which guarantees a kept stop), or fly `i` atop an already-used skip.

Initialize `f0 = a[0]`, `f1 = -inf` (no skip is possible before any stop), `ans = f0`. Then loop from
`i = 1`, updating `ans = max(ans, f0_i, f1_i)`. Answer is `ans`.

**Pitfalls.**
1. *Wrong base case / sign (the load-bearing one).* Seeding `ans = 0` (or letting `f0` reset to `0`)
   smuggles in the forbidden empty mission, so an all-negative corridor like `[-7]` wrongly returns
   `0` instead of `-7`. Seed `ans` from the always-legal singleton `a[0]` and never admit a bare `0`
   into the maximum.
2. *Phantom skip.* Initializing `f1 = 0` invents a "skipped everything, kept nothing" block worth `0`,
   illegal because a skip requires a kept stop. Start `f1` at `-inf`; on `[-3,-5]` the `0` seed makes
   the answer `0` instead of `-3`.
3. *Update order.* `f1_i` reads the **old** `f0`; compute both new values into temporaries before
   assigning, or `f1` will read the just-updated `f0` and model "skip stop `i` while also flying it".
4. *Overflow.* With `n` up to `2*10^5` and `|a[i]|` up to `10^9`, totals reach `~2*10^14`; use
   `long long`. An `int` is a silent wrong answer on large tests. The `LLONG_MIN/4` sentinel is only
   ever read in a `max` or given a single `+a[i]`, so it cannot underflow.

**Edge cases.** `n = 1` -> `a[0]` (no skip possible); all-negative -> least-negative element (e.g.
`[-2,-5,-3,-1] -> -1`); a single zero -> `0` (a real stop, not the empty mission); a deep negative
bridging two positive runs is jumped by the skip (`[2,-50,3,4,-1,5,-100] -> 13`); all-positive ->
whole-array sum, since removing a positive only loses charge.

**Complexity.** `O(n)` time, `O(1)` extra space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // f0 = best sum of a contiguous non-empty window ending here, no skip used.
    // f1 = best sum ending here with exactly one interior stop skipped (>=1 kept).
    const long long NEG = LLONG_MIN / 4;
    long long f0 = a[0];          // window {0}, nothing skipped
    long long f1 = NEG;           // cannot skip anything yet (need a kept element)
    long long ans = f0;           // best so far must come from a non-empty pick

    for (int i = 1; i < n; i++) {
        long long nf1 = max(f0, f1 + a[i]); // skip i now (extend f0) OR keep i atop f1
        long long nf0 = max(a[i], f0 + a[i]); // start fresh at i, or extend f0
        f0 = nf0;
        f1 = nf1;
        ans = max(ans, max(f0, f1));
    }

    cout << ans << "\n";
    return 0;
}
```
