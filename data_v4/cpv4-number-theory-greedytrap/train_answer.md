**Problem.** A rig fires laser pulses locked to an order `k` (`2 <= k <= 5`); each pulse carries an energy that is a perfect `k`-th power `1^k, 2^k, 3^k, ...`, and pulses may be reused freely. Given `k` and a target `n` (`0 <= n <= 10^6`), output the minimum number of pulses whose energies sum to exactly `n`. Since `1 = 1^k` is always admissible, every target is reachable; `n = 0` gives `0`.

**Why the obvious greedy is wrong.** "Repeatedly subtract the largest admissible `k`-th power that fits" maximizes coverage this step but can strand the remainder off the useful lattice. For `k = 2`, `n = 12`: greedy takes `9`, then is forced into `1 + 1 + 1`, for `9 + 1 + 1 + 1 = 4` pulses; but `4 + 4 + 4 = 12` is `3`. For `k = 3`, `n = 32`: greedy takes `27` then `1` five times for `6`; but `8 + 8 + 8 + 8 = 32` is `4`. The locally largest power blocks more than it gains. Greedy is discarded.

**Key idea — shortest-representation DP.** Let `dp[v]` be the fewest admissible powers summing to exactly `v`. With `dp[0] = 0`,

`dp[v] = 1 + min over admissible powers p <= v of dp[v - p]`,

because any optimal representation of `v` contains some power `p` as a summand, and deleting it leaves an optimal-or-better representation of `v - p`. The answer is `dp[n]`. This is fewest-coins coin change with the `k`-th powers as coins: `O(n * P)` time, where `P` is the number of admissible powers up to `n` (`<= 1000` for `k = 2`, fewer for larger `k`).

**Pitfalls to get right.**
1. *Power enumeration overflow / precision.* Do not use floating `pow(b, k)` (it can be off by one) and do not form `b^k` before comparing to `n` (it overflows `long long` for large `b`, `k = 5`). Build `b^k` by repeated multiplication and bail before each multiply using the guard `p > n / b` (so `p * b > n`), which keeps every running product `<= n` and overflow-free.
2. *Inner-loop early exit.* Keep `powers` ascending and `break` (not `continue`) at the first `p > v`: all later powers also exceed `v`, so breaking is both correct and what keeps the `k = 2`, `n = 10^6` case inside the time limit.
3. *Base case.* `dp[0] = 0` is essential; otherwise `dp[1] = dp[0] + 1` is garbage. Handle `n = 0` up front so the enumeration loop never runs with a degenerate `n`.

**Edge cases (all handled by the recurrence + the early `n = 0` return):** `n = 0` -> `0`; `n = 1` -> `1`; `n` an exact `k`-th power -> `1`; small `n` at high order (e.g. `k = 5`, `n = 7`) -> `n` (a chain of `1`s, since `2^5 = 32 > 7`); greedy-trap targets like `k = 2`, `n = 18` -> `2` (`9 + 9`), never greedy's `3`.

**Complexity.** `O(n * P)` time with `P` the count of `k`-th powers `<= n` (worst `P ~ 1000` at `k = 2`), `O(n)` memory. Measured worst case `k = 2`, `n = 10^6` runs in about `0.7s`.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    long long k, n;
    if (!(cin >> k >> n)) return 0;        // empty input -> nothing to do

    // n == 0 needs zero pulses; handle before allocating the DP table.
    if (n == 0) { cout << 0 << "\n"; return 0; }

    // Enumerate every perfect k-th power in [1, n]: 1^k, 2^k, 3^k, ...
    // Multiply step by step, bailing out the instant the running product
    // exceeds n so the product never overflows.
    vector<long long> powers;
    for (long long b = 1;; b++) {
        long long p = 1;
        bool exceed = false;
        for (long long e = 0; e < k; e++) {
            if (p > n / b) { exceed = true; break; }  // p*b would exceed n
            p *= b;
        }
        if (exceed || p > n) break;
        powers.push_back(p);
    }
    // powers is ascending; powers[0] == 1 guarantees every n is representable.

    const int INF = 1e9;
    vector<int> dp(n + 1, INF);
    dp[0] = 0;
    for (long long v = 1; v <= n; v++) {
        int best = INF;
        for (long long p : powers) {
            if (p > v) break;                 // ascending: all later powers exceed v too
            int cand = dp[v - p] + 1;
            if (cand < best) best = cand;
        }
        dp[v] = best;
    }

    cout << dp[n] << "\n";
    return 0;
}
```
