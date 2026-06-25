**Problem.** Cross a river on stones `0..n-1`. A leap moves forward by one or two stones. The first
landing off the near bank must be on stone `0` or stone `1`; from stone `i` you land next on `i+1`
or `i+2`; the final leap to the far bank is free and may be taken from stone `n-1` or stone `n-2`.
Landing on stone `i` costs `c[i]` (which may be negative). Read `n` and the costs from stdin and
print the minimum total cost of the stones you land on. For `n = 0` the answer is `0`.

**Why the obvious greedy is wrong.** "Land on the cheaper of stone `0` / stone `1`, then from each
stone hop to the cheaper of the next two" fails because the `+1 / +2` reach is global: each landing
fixes the next two options. On `c = [1, 8, 9, 5]` greedy goes `0 -> 1 -> 3` and pays `1 + 8 + 5 =
14`, but `0 -> 2` (paying `1 + 9 = 10`) lands on stone `2 = n-2`, from which a single free leap
clears the bank — two landings instead of three. Choosing the cheaper *next* stone ignored the
costlier *future* it committed to. Greedy is discarded.

**Key idea — linear prefix DP.** Let `dp[i]` be the minimum total cost over all legal crossings that
end *standing on stone `i`*. To stand on stone `i` (`i >= 2`) you just leapt from stone `i-1` or
stone `i-2`, then paid `c[i]`:

- `dp[i] = min(dp[i-1], dp[i-2]) + c[i]`

The two ends carry the rules:

- *First landing* is stone `0` or stone `1` directly:
  `dp[0] = c[0]`, and `dp[1] = min(c[1], dp[0] + c[1])` (land on `1` first, or step `0 -> 1`).
- *Final leap* is free from stone `n-1` or stone `n-2`:
  `answer = min(dp[n-1], dp[n-2])` (drop the `dp[n-2]` term when `n = 1`).

With negative costs the same `min`-recurrence automatically prefers chaining through more stones (a
`+1` step touches every stone), so nothing special is needed for the all-negative case.

**Pitfalls.**
1. *Finishing only from the last stone.* Writing `answer = dp[n-1]` over-charges any optimum that
   leaps off from stone `n-2`. On `[1, 8, 9, 5]` this returns `14` instead of `10`. Fix:
   `answer = min(dp[n-1], dp[n-2])`.
2. *Forbidding the direct first landing.* Writing `dp[1] = dp[0] + c[1]` forces a step `0 -> 1` and
   bans landing on stone `1` first. On `[9, 1]` it returns `9` instead of `1`. Fix:
   `dp[1] = min(c[1], dp[0] + c[1])`.
3. *Overflow.* With `n` up to `2*10^5` and `|c[i]|` up to `10^9`, the sum reaches `~2*10^14`; use
   `long long`. An `int` is a silent wrong-answer on large tests.

**Edge cases.** `n = 0` -> `0` (special-cased before indexing `dp`); `n = 1` -> `c[0]` (must land on
the lone stone); `n = 2` -> `min(c[0], c[1])`; all-negative -> land on every stone (the recurrence
does this automatically).

**Complexity.** `O(n)` time, `O(n)` space (foldable to `O(1)`).

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;            // n = 0: already across, cost 0
    vector<long long> c(n);
    for (auto &x : c) cin >> x;

    // dp[i] = minimum total stamina to be standing on stone i, having legally
    // arrived there from the near bank. You may first land on stone 0 or stone 1.
    // From stone j you may have come from stone j-1 or stone j-2.
    // Far bank is reachable from stone n-1 or stone n-2 at no extra cost.
    if (n == 0) { cout << 0 << "\n"; return 0; }

    vector<long long> dp(n);
    dp[0] = c[0];                          // first leap lands on stone 0
    if (n >= 2) dp[1] = min(c[1], dp[0] + c[1]); // first leap onto 1, or step 0->1
    for (int i = 2; i < n; i++) {
        dp[i] = min(dp[i - 1], dp[i - 2]) + c[i];
    }

    // reach far bank from stone n-1 or stone n-2
    long long ans = dp[n - 1];
    if (n >= 2) ans = min(ans, dp[n - 2]);

    cout << ans << "\n";
    return 0;
}
```
