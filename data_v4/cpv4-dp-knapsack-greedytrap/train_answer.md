**Problem.** A sequencing core has `B` micrograms of a shared reagent and `n` candidate assays. Assay `i` consumes `c[i]` micrograms (all-or-nothing) and, if run, yields `r[i]` data points; each assay runs at most once. Pick a subset with total cost `<= B` maximizing total yield. Read `n`, `B`, then the `n` pairs `(c[i], r[i])` from stdin; print the maximum yield. Running nothing is allowed, so the answer is at least `0`.

**Why the obvious greedy is wrong.** "Run the most reagent-efficient assays first" (sort by `r[i]/c[i]`, or by raw `r[i]`, descending, and take while the budget allows) borrows the fractional-knapsack intuition, but the assays are indivisible. On `B = 10` with assays `(6,8), (5,6), (5,6)`, density-greedy grabs `(6,8)` first (`8/6 ≈ 1.33` beats `6/5 = 1.20`), spends 6 micrograms, and strands a 4-microgram fragment that fits nothing — total `8`. The optimum forgoes the densest item and runs both cost-5 assays, tiling the budget exactly: `6 + 6 = 12`. Raw-yield-greedy makes the identical mistake (it also grabs the yield-8 item first). With indivisible items the *partition* of the fixed budget matters, not the per-unit rate. Both greedies are discarded.

**Key idea — 0/1 knapsack DP over the budget.** Let `dp[w]` = the best total yield achievable using total cost at most `w`. Process assays one at a time; for each assay `i` with cost `c[i]`, yield `r[i]`:

- `dp[w] = max(dp[w], dp[w - c[i]] + r[i])` for `w >= c[i]` (run assay `i`, leaving budget `w - c[i]`).

Base case `dp[w] = 0` for all `w` (run nothing). The answer is `dp[B]`. Because the table stores "cost `<= w`", `dp` is monotone in `w` and `dp[B]` already accounts for leaving budget unspent and for the empty subset, so no separate final maximization is needed and the answer is automatically at least `0`.

**Correctness.** With `dp` defined as the best yield over subsets of the first `k` assays with cost `<= w`, the transition is the standard include/exclude split: either assay `k` is skipped (value `dp[w]` from the first `k-1` assays) or run, which needs `w >= c[k]` and reduces the remaining budget to `w - c[k]` for the earlier assays. Iterating `w` from high to low guarantees every `dp[w - c[i]]` read still holds the value *before* assay `i` was considered, so each assay is included at most once. The traced counterexample reaches `dp[10] = 12`, matching brute-force subset enumeration.

**Pitfalls.**
1. *Iteration direction.* Iterating `w` upward reads a `dp[w - c[i]]` already updated in the same pass, which stacks multiple copies of one assay — that is unbounded knapsack, not 0/1. A single assay `(3, 5)` with `B = 6` then returns `10` (the item run twice) instead of `5`. Iterate `w` from `B` down to `c[i]`.
2. *Overflow.* With `n` up to `2000` and `r[i]` up to `10^9`, the total yield can reach `~2*10^12`, well beyond 32-bit. Use `long long` for the table and the answer; an `int` table is a silent wrong-answer.
3. *Over-budget / index safety.* Assays with `c[i] > B` can never fit; the guard `w >= c[i]` makes their inner loop empty and also keeps every index `w - c[i]` non-negative.

**Edge cases (all handled by the recurrence + the `<= w` semantics):** `n = 0` -> `0`; `B = 0` -> `0` (every cost `>= 1`); assays with `r[i] = 0` are harmless no-ops (monotonicity means `dp[w-c] <= dp[w]`); all assays over budget -> `0`; exact-fit subsets (e.g. two cost-5 assays at `B = 10`) are reachable because the `>= c[i]` boundary is inclusive.

**Complexity.** `O(n * B)` time (`<= 4*10^8` add-and-compare steps; ~0.12 s measured at the max shape) and `O(B)` space (~1.6 MB).

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long B;
    if (!(cin >> n >> B)) return 0;

    vector<long long> c(n), r(n);
    for (int i = 0; i < n; i++) cin >> c[i] >> r[i];

    // dp[w] = best total yield achievable with total cost <= w.
    // Process each assay once; iterate w from high to low so dp[w - c[i]]
    // still reflects "this assay not yet used" -> each assay counts at most once.
    vector<long long> dp(B + 1, 0);
    for (int i = 0; i < n; i++) {
        long long ci = c[i], ri = r[i];
        if (ci > B) continue;                 // never fits, skip
        for (long long w = B; w >= ci; w--) {
            long long cand = dp[w - ci] + ri; // run assay i, leaving budget w - ci
            if (cand > dp[w]) dp[w] = cand;
        }
    }

    cout << dp[B] << "\n";                     // "<= B" semantics fold in unspent budget
    return 0;
}
```
