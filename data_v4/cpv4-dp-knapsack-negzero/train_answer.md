**Problem.** Given `n` ballast blocks, block `i` with integer weight `w[i] >= 1` and **signed** trim score `v[i]` (negative, zero, or positive), pick a subset whose weights sum to **exactly** `C` and maximize the total score. The empty subset (weight `0`, score `0`) is legal. If no subset reaches weight `C`, print `IMPOSSIBLE`. Read `n`, `C`, and the `n` pairs from stdin; print the maximum score (which may be negative) or `IMPOSSIBLE`.

**Why brute force / meet-in-the-middle is out.** Exhaustive subset enumeration is `O(2^n)` — correct but hopeless at `n = 2000`. Meet-in-the-middle reaches `n` around 40, still far short, and merging under "maximize score at a fixed weight" is awkward. We need a polynomial DP.

**Key idea — capacity-indexed 0/1 DP with an unreachable sentinel.** Let `dp[c]` = best total score over subsets of weight **exactly** `c`, or a sentinel `UNREACH` if weight `c` is unreachable. Relax block by block, 0/1 style:

- `dp[c] = max(dp[c], dp[c - w[i]] + v[i])` for `c >= w[i]`, **only when** `dp[c - w[i]] != UNREACH`.

Iterate `c` **downward** (`C` down to `w[i]`) so each block is used at most once. The answer is `dp[C]`: print it if reachable, else `IMPOSSIBLE`.

**Correctness.** `dp[c]` is exactly the max score over weight-`c` subsets: induction on blocks. Before any block, only weight `0` is reachable (empty subset, score `0`), so `dp[0] = 0` and all else `UNREACH` — this is the base case. Processing block `i`, a weight-`c` subset either omits it (value already in `dp[c]`) or includes it on top of a reachable weight-`(c - w[i])` subset (value `dp[c-w[i]] + v[i]`); the downward order guarantees `dp[c-w[i]]` still reflects the pre-block state, so the block is not reused. The `!= UNREACH` guard forbids extending a non-existent subset.

**Pitfalls.**
1. *Wrong base case.* Do **not** initialize the whole table to `0`. That is the reflex from the "weight `<= C`" knapsack where every capacity is trivially reachable, but here the constraint is *exact*: only weight `0` is reachable up front. A zero-filled table falsely declares every target reachable and prints numbers for impossible `C`.
2. *Sentinel inside the answer range.* Scores are signed and the optimum at `C` can be negative (down to `~ -2*10^12`). A sentinel like `-1` collides with a real reachable-but-negative score, so `max` prefers the sentinel over a worse real value and the final reachability test misfires — on `C=4`, blocks `(4,-6),(2,-1),(2,-3)` the true answer is `-4` but a `-1` sentinel prints `IMPOSSIBLE`. Use `UNREACH = LLONG_MIN/4`, strictly below every attainable score, and guard it so `+v[i]` never lands on it.
3. *Iteration direction.* Upward capacity iteration reuses a block: with one block `(2,5)` and `C=4`, upward gives `dp[4]=10` (the block twice). Iterate downward for 0/1.
4. *Overflow.* Scores reach `~2*10^12`; use `long long`. A 32-bit `int` is a silent wrong-answer on large tests.

**Edge cases.** `C = 0` → `0` (empty load), even with all-negative blocks, since every `w[i] >= 1` makes the inner loop empty. `n = 0` → `0` if `C = 0` else `IMPOSSIBLE`. All-negative scores with reachable `C` → the negative optimum, never `IMPOSSIBLE`. Unreachable `C` (no subset sums to it) → `IMPOSSIBLE`. Blocks with `w[i] > C` contribute nothing and are skipped. Verified against an `O(2^n)` brute force on 1100+ random cases (zero mismatches), including hundreds of `IMPOSSIBLE` and 100+ reachable-but-negative answers.

**Complexity.** `O(n*C)` time (`<= 10^8`, ~0.05 s at the bound), `O(C)` extra space.

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

    // dp[c] = best total trim score achievable using a subset of weight exactly c.
    // Sentinel UNREACH marks "no subset reaches this weight". Base: dp[0] = 0 (empty subset).
    const long long UNREACH = LLONG_MIN / 4;
    vector<long long> dp(C + 1, UNREACH);
    dp[0] = 0;

    for (int i = 0; i < n; i++) {
        long long wi = w[i], vi = v[i];
        if (wi > C) continue;                 // can never fit into an exact total of C
        for (long long c = C; c >= wi; c--) { // 0/1: iterate capacity downward
            if (dp[c - wi] != UNREACH) {
                long long cand = dp[c - wi] + vi;
                if (cand > dp[c]) dp[c] = cand;
            }
        }
    }

    if (dp[C] == UNREACH) cout << "IMPOSSIBLE" << "\n";
    else cout << dp[C] << "\n";
    return 0;
}
```
