**Problem.** A planetarium has `n` projector modules. Module `i` draws `w[i]` watts and gives `b[i]`
units of brightness. The single power supply delivers at most `C` watts. Switch on any subset of
modules whose total wattage is `<= C` to maximize total brightness; print that maximum. This is 0/1
knapsack: weights `w`, values `b`, capacity `C`, each module used at most once. Read `n`, `C`, then
the `n` pairs from stdin; output the maximum total brightness (`0` if nothing fits).

**Key idea — capacity DP.** Let `dp[c]` = the best total brightness with total wattage at most `c`.
Start `dp[c] = 0` for all `c` (empty subset). Fold modules in one at a time; for module `(w[i], b[i])`,

  `dp[c] = max( dp[c], dp[c - w[i]] + b[i] )`  for every `c >= w[i]`.

Iterate `c` from `C` **down to** `w[i]`, so `dp[c - w[i]]` still holds the value *before* this module
was considered — that enforces "each module at most once" (0/1). The answer is `dp[C]`. With
`C <= 4000` and `n <= 2000` this is `O(n * C) ≈ 8 * 10^6` operations, far inside 1 second.

**Pitfalls.**
1. *Overflow — the central trap.* With `n` up to `2000` and `b[i]` up to `10^9`, the answer can reach
   `2 * 10^12`, about `931x` past `INT_MAX (2147483647)`. An `int` DP passes every small sample and
   then silently wraps: trace `5` modules of `(w, b) = (1, 10^9)` with `C = 5` (true answer
   `5 * 10^9`). The `int` table climbs to `2 * 10^9`, then the next candidate `3 * 10^9` wraps to
   `-1294967296`, which loses the `if (cand > dp[c])` comparison, so `dp[C]` freezes at `2000000000`
   instead of `5000000000`. Make `dp` and all partial sums `long long`.
2. *Iteration direction.* Looping `c` upward turns this into unbounded knapsack (a module can refit
   into the leftover capacity and be taken twice). On `(2, 3)` with `C = 4`, upward gives `6`,
   downward gives the correct `3`.
3. *Module heavier than `C`.* Guard with `if (w[i] > C) continue;` (and the loop bound `c >= w[i]` is
   already empty in that case), so it contributes nothing.

**Edge cases.** `C = 0` (or every `w[i] > C`): no module fits, answer `0` (empty subset). Single
module that fits: its brightness. Leftover wattage is fine — `dp[C]` means "wattage at most `C`," so
the all-off subset is the base case and unused watts cost nothing. Reading guarded by
`if (!(cin >> n >> C)) return 0;`.

**Complexity.** `O(n * C)` time, `O(C)` extra space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long C;
    if (!(cin >> n >> C)) return 0;

    vector<long long> w(n), b(n);
    for (int i = 0; i < n; i++) cin >> w[i] >> b[i];

    // dp[c] = maximum total brightness using a subset whose total wattage is exactly <= c.
    // Brightness sums can reach n * 1e9 = 2e12, so dp MUST be 64-bit.
    vector<long long> dp(C + 1, 0);
    for (int i = 0; i < n; i++) {
        long long wi = w[i], bi = b[i];
        if (wi > C) continue;                 // module can never fit
        for (long long c = C; c >= wi; c--) { // 0/1 knapsack: iterate capacity downward
            long long cand = dp[c - wi] + bi;
            if (cand > dp[c]) dp[c] = cand;
        }
    }

    cout << dp[C] << "\n";
    return 0;
}
```
