**Problem.** A row of `n` plant beds, bed `i` needing `a[i] >= 0` liters, must be cut into `k` contiguous blocks (one per watering robot; empty blocks allowed when `k > n`). A robot's time is its block's water sum, robots run in parallel, so the greenhouse finishes at the *maximum* block sum. Minimize that maximum and print it. Read `n`, `k`, and the values from stdin.

**Why the obvious greedy is wrong.** "Balance every block near the fair share `S/k`" optimizes the wrong quantity: it minimizes spread, not the maximum. On `[7, 1, 2, 9, 2, 6]`, `k = 3` (`S = 27`, share `9`) the balance greedy cuts `[7,1] | [2,9] | [2,6]` with max `11`, but the deliberately *unbalanced* `[7,1,2] | [9] | [2,6]` has max `10`. Worse, the average can be infeasible: on `[2,3,5,8,1,1,4]`, `k = 3` the share is `8`, yet filling blocks up to `8` needs four blocks because the lone `8` plus anything overflows — the true answer is `10`. Minimizing the max sometimes demands unbalanced blocks, so the average greedy is discarded.

**Key idea — binary search on the finish time.** Fix a candidate maximum `T` and ask the monotone decision question: can the row be cut into at most `k` contiguous blocks each summing `<= T`? The fewest-blocks greedy answers it exactly — extend the current block while it stays `<= T`, cut the moment the next bed would overflow, and reject `T` outright if any single `a[i] > T`. Feasibility is monotone in `T` (more budget never hurts), so the smallest feasible `T` is found by bisection on `[max(a[i]), S]`:

- lower bound `max(a[i])` — some block must hold the biggest bed;
- upper bound `S = sum(a)` — one block holding everything is always feasible since `k >= 1`.

Standard lower-bound bisection: while `lo < hi`, test `mid = lo + (hi-lo)/2`; if feasible set `hi = mid`, else `lo = mid + 1`. The loop ends with `lo` = the minimum feasible finish time.

**Correctness.** The greedy feasibility check uses the minimum number of blocks for a given `T` (extending greedily can never force a later block to hold more — an exchange argument), so it answers "achievable with `<= k` blocks?" correctly. Feasibility is monotone, so bisection converges to the least `T` with a valid `k`-block partition, which is exactly the minimized maximum.

**Pitfalls.**
1. *Block-count off-by-one.* The number of blocks is "one initial block plus one per overflow." Starting the counter at `0` undercounts by one and flips the verdict — e.g. on `[3,3,3]`, `k = 2` it reports `2` blocks (feasible) when `[3]|[3]|[3]` plainly needs `3`. Start `blocks = 1`.
2. *Lower bound below `max(a[i])`.* A single bed can exceed the average, so the answer floor is `max(a[i])`, not `S/k`; the `a[i] > T` guard inside feasibility enforces it.
3. *Overflow.* With `n` up to `2*10^5` and `a[i]` up to `10^9`, `S` reaches `~2*10^14`; use `long long` for the total, accumulators, bounds, and answer. Use `mid = lo + (hi-lo)/2`, never `(lo+hi)/2`.

**Edge cases.** `n = 0` → `0` (handled explicitly); all zeros → `0`; `k = 1` → whole sum `S`; `k >= n` → `max(a[i])` (every bed alone); a lone bed far above the average sets the floor and is handled by the `lo = max(a[i])` bound plus the per-bed guard.

**Complexity.** `O(n log S)` time, `O(n)` space for the input (`O(1)` working memory beyond it).

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long k;
    if (!(cin >> n >> k)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // Feasibility: can we partition a[0..n-1] into at most k contiguous blocks,
    // each with block-sum <= T?  Greedy: extend the current block while the running
    // sum stays <= T; when it would exceed, cut (start a new block at this bed).
    auto feasible = [&](long long T) -> bool {
        long long cur = 0;
        long long blocks = 1;            // we always use at least one block
        for (int i = 0; i < n; i++) {
            if (a[i] > T) return false;  // a single bed already exceeds T
            if (cur + a[i] > T) {        // start a new block at bed i
                blocks++;
                cur = a[i];
                if (blocks > k) return false;
            } else {
                cur += a[i];
            }
        }
        return blocks <= k;
    };

    // Lower bound: the largest single bed (some block must hold it).
    // Upper bound: the total (one block holds everything).
    long long lo = 0, hi = 0;
    for (long long x : a) { lo = max(lo, x); hi += x; }

    // Empty greenhouse: no beds, finishing time is 0.
    if (n == 0) { cout << 0 << "\n"; return 0; }
    // If k >= n every bed can be its own block, so the answer is the max bed.
    // (The binary search handles this too, but it is a clean sanity anchor.)

    while (lo < hi) {
        long long mid = lo + (hi - lo) / 2;
        if (feasible(mid)) hi = mid;
        else lo = mid + 1;
    }

    cout << lo << "\n";
    return 0;
}
```
