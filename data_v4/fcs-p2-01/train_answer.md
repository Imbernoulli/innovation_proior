**Problem.** Given `n` coin denominations and a target `S`, with an unlimited supply of each
denomination, form `S` exactly using as few coins as possible; print that minimum count, or `-1`
if `S` cannot be formed. Read `n`, `S`, and the `n` denominations from stdin; print one integer.
Constraints: `1 <= n <= 100`, `0 <= S <= 10^6`, `1 <= c[i] <= 10^6`.

**Why the obvious greedy is wrong.** "Repeatedly take the largest denomination `<= remaining`" is the
instinctive move, and it *is* optimal for canonical currency systems like `{1, 5, 10, 25}`. But the
denominations here are arbitrary, and for an arbitrary set greedy can strand a remainder the set fills
inefficiently:
- `{1, 3, 4}`, `S = 6`: greedy takes `4 + 1 + 1 = 3` coins; optimal is `3 + 3 = 2`.
- `{1, 5, 6, 9}`, `S = 11`: greedy takes `9 + 1 + 1 = 3` coins; optimal is `5 + 6 = 2`.

Grabbing the largest coin is a locally tempting, globally worse choice, and it is structural rather
than a patchable corner. Greedy is discarded.

**Key idea — bottom-up DP over sums.** Let `dp[s]` be the minimum number of coins summing to exactly
`s`, with `dp[s] = INF` when `s` is unreachable. Then

- `dp[0] = 0` (zero coins make the empty sum),
- for `s >= 1`: `dp[s] = 1 + min over denominations c with c <= s of dp[s - c]`, taking only
  reachable predecessors (`dp[s - c] != INF`).

An exchange argument proves this: any optimal multiset for `s` contains some coin `c`; deleting it
leaves an optimal multiset for `s - c` with one fewer coin. Filling `dp` in increasing order of `s`
makes every predecessor `s - c < s` final before it is read, and — because we relax *forward* over
increasing `s` — each denomination can be reused freely, which is exactly the unlimited-supply
(unbounded knapsack) order. The answer is `dp[S]`, printed as `-1` if it is still `INF`.

**Two pitfalls to get right.**
1. *Sentinel overflow.* Do not add `1` to an `LLONG_MAX` sentinel: `LLONG_MAX + 1` overflows to a
   negative, and `min` then latches onto the garbage. Use a finite sentinel well below `LLONG_MAX`
   (here `4e18`) *and* guard each relaxation on `dp[s - c] != INF` so unreachable sub-sums never
   propagate. (A trace of `{4}`, `S = 6` printing a negative instead of `-1` exposes exactly this.)
2. *Unreachable / `S = 0`.* Map a final `INF` to `-1`; `S = 0` correctly yields `0` because the loop
   never runs and `dp[0] = 0`.

**Edge cases (all handled by the recurrence + the `INF` guard):** `S = 0` -> `0`; coins all larger
than `S` -> `-1`; impossible targets (e.g. `{2, 4}`, `S = 3`) -> `-1`; duplicate denominations are
harmless; a single denomination dividing `S` works.

**Complexity.** `O(S * n)` time, `O(S)` space. At `S = 10^6`, `n = 100` that is `~10^8` simple
`long long` relaxations, about `0.1` s in practice and well inside a `2` s limit; the `dp` array is
`~8` MB, well inside `256` MB.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long S;
    if (!(cin >> n >> S)) return 0;        // empty input -> nothing to do
    vector<long long> c(n);
    for (auto &x : c) cin >> x;

    // dp[s] = minimum number of coins to make sum s exactly; INF if unreachable.
    // Unbounded supply of each denomination, so each coin relaxes every larger sum.
    const long long INF = (long long)4e18;
    vector<long long> dp((size_t)S + 1, INF);
    dp[0] = 0;                              // zero coins make sum 0
    for (long long s = 1; s <= S; s++) {
        for (int i = 0; i < n; i++) {
            long long v = c[i];
            if (v <= s && dp[s - v] != INF && dp[s - v] + 1 < dp[s]) {
                dp[s] = dp[s - v] + 1;
            }
        }
    }

    cout << (dp[S] == INF ? -1 : dp[S]) << "\n";
    return 0;
}
```
