**Problem.** A submersible has a mass budget `C`. There are `n` crates; crate `i` has mass `w[i]` and value `v[i]`, and is loaded whole or not at all. Pick a subset with total mass `<= C` maximizing total value; the empty load is allowed, so the answer is at least `0`. Read `n`, `C`, then the `n` pairs from stdin and print the maximum value. Constraints: `n <= 1000`, `C <= 2000`, `w[i] <= 2000`, `v[i] <= 10^9`.

**Why the obvious greedy is wrong.** "Sort by value/mass ratio and load while it fits" is optimal for the *fractional* knapsack, which is exactly why it is tempting — and it fails here because crates are indivisible. On `C = 10` with crates `(6, 10), (5, 7), (5, 7)`, the best-ratio crate is the first (`10/6 ≈ 1.67`); greedy loads it, strands the remaining `4` units of budget (no other crate fits in `4`), and stops at value `10`. But loading the two mass-5 crates fills the hold exactly for `7 + 7 = 14`. A high-ratio crate can waste more budget than its value justifies once you cannot slice it. Greedy is discarded.

**Key idea — capacity DP.** Let `dp[c]` be the best total value of a subset whose mass is *exactly* `c`, with `-1` meaning "mass `c` unreachable" (kept distinct from a real reachable value of `0`). Initialize `dp[0] = 0`. Fold crates in one at a time; for crate `(w, v)`,

`dp[c] = max( dp[c],  dp[c - w] + v )`  for reachable `dp[c - w]`.

The answer is `max(0, max_c dp[c])`.

**Pitfalls to get right.**
1. *Sweep direction.* Update capacity **downward** (`c = C .. w`). A forward sweep reads `dp[c - w]` after it was already updated this pass, loading the same crate twice — that is the *unbounded* knapsack. (A trace of a single `(5, 7)` crate returning `14` instead of `7` exposes exactly this.)
2. *Overflow.* With `n = 1000` and `v[i]` up to `10^9`, the total value can reach `~10^12`; accumulate values in `long long`. An `int` is a silent wrong-answer on large tests (the max-constraint answer `25628973081` already exceeds `2^31`).
3. *Reachability sentinel.* Use `-1` for "unreachable", not `0`, so an empty/zero-value mass is not confused with an impossible one; never add `v` to the sentinel (guard with `!= -1`).

**Edge cases (all handled by the recurrence + guards):** `n = 0` and `C = 0` give `0`; a crate with `w > C` is skipped by an explicit guard; zero-mass crates are loaded once each (the downward sweep visits `c` once per crate, so no double-count); zero-value crates never inflate the answer.

**Complexity.** `O(n * C)` time (`<= 2 * 10^6` cell updates), `O(C)` space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long C;
    if (!(cin >> n >> C)) return 0;
    vector<long long> w(n), v(n);
    for (int i = 0; i < n; i++) cin >> w[i] >> v[i];

    // dp[c] = best total value using a chosen subset with total weight EXACTLY c,
    // or -1 if c is unreachable. dp[0] = 0 (empty subset). 0/1 knapsack: iterate
    // capacity downward so each crate is used at most once.
    const long long NEG = -1;
    vector<long long> dp(C + 1, NEG);
    dp[0] = 0;
    for (int i = 0; i < n; i++) {
        if (w[i] > C) continue;                 // crate alone exceeds the hold
        for (long long c = C; c >= w[i]; c--) {
            if (dp[c - w[i]] != NEG) {
                long long cand = dp[c - w[i]] + v[i];
                if (cand > dp[c]) dp[c] = cand;
            }
        }
    }

    long long best = 0;                         // empty subset has value 0
    for (long long c = 0; c <= C; c++)
        if (dp[c] != NEG && dp[c] > best) best = dp[c];

    cout << best << "\n";
    return 0;
}
```
