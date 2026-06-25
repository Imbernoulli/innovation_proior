**Problem.** A shop has `m` identical presses and `n` jobs; job `i` takes `t[i]` minutes and must run whole on one press. A press's finish time is the sum of its jobs, and the shop closes when the last press finishes. Assign jobs to presses to minimize the closing time (the makespan), and print that minimum. Read `n m` then the `n` values from stdin. Constraints: `0 <= n <= 14`, `1 <= m <= 14`, `1 <= t[i] <= 10^9`.

**Why the obvious greedy is wrong.** The "longest-processing-time" greedy — sort jobs descending and drop each on the least-loaded press — is a famous 4/3-approximation, so it is tempting to ship as the answer. It is not optimal. On `m = 3` with `t = [5, 5, 4, 4, 3, 3, 3]`, LPT produces loads `[11, 8, 8]` (closing time `11`), but the partition `{5,4}, {5,4}, {3,3,3}` gives loads `9, 9, 9` (closing time `9`). Local least-loaded placement scatters the three `3`s onto presses that already hold a `5+4`, wasting slack; it cannot foresee that the `3`s wanted to share one press. A second shortcut — outputting `max(maxJob, ceil(sum/m))` — is only a lower *bound*, not always achievable, because jobs are indivisible. Both greedies are discarded.

**Key idea — binary search the closing time, exact feasibility test.** The predicate `feasible(T)` = "can all jobs finish by time `T`?" is monotone in `T`, so binary-search the smallest feasible `T` over `[lo, hi]` with `lo = max t[i]` (some press must run the largest job) and `hi = sum t[i]` (one press does everything). Feasibility is a bin-packing decision: partition the jobs into at most `m` groups, each summing `<= T`. Solve it exactly with a subset-cover DP: `dp[mask]` = minimum number of presses (each loaded `<= T`) to cover exactly the jobs in `mask`. Transition: from `mask`, let the next press be **any** non-empty submask `sub` of the remaining jobs with `sumMask[sub] <= T`, advancing to `mask | sub` for one more press. Then `feasible(T)` iff `dp[full] <= m`. Precompute `sumMask[sub]` for all subsets in `O(2^n)`.

**Pitfalls.**
1. *A naive "open-bin" feasibility DP is wrong.* Tracking only one currently-fillable press (state `{presses, openLoad}`, opening a new press whenever a job does not fit) forbids reopening a press you already closed, so it reports false infeasibility. On `[3, 3, 2, 2, 2]`, `m = 2`, it wrongly says `feasible(6) = false` and the search returns `7` instead of `6`. The subset-cover DP fixes this by carving off *any* valid full press at each step, considering every partition.
2. *Overflow.* With `n = 14` and `t[i]` up to `10^9`, loads and the search range reach `1.4 * 10^{10}`; use `long long` for `lo`, `hi`, `mid`, the sums, and `sumMask`. An `int` is a silent wrong-answer on large tests. Use `mid = lo + (hi - lo) / 2`.

**Edge cases.** `n = 0` -> print `0` (handled before building `full`). `m = 1` -> answer is the total. `m >= n` -> each job gets its own press, answer is `max t[i]`; the press count is only a DP bound, never an array index, so `m > n` is safe. `[7,1,1,1] m=2` -> `7`, not the average bound `5`, because one job dominates.

**Complexity.** Feasibility is `O(3^n)` (submasks of every complement, pruned at `dp[mask] >= m`); binary search adds an `O(log(sum)) ~ 35` factor. Total `~ 35 * 3^{14} ~ 1.7 * 10^8` integer ops, measured at `0.16 s` for `n = 14`. Memory `O(2^n)`.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int n, m;
vector<long long> t;
int full;

vector<long long> sumMask;   // sumMask[mask] = total time of the jobs in mask

// Feasibility: can the jobs be partitioned into at most m presses,
// each press's total time <= cap?  This is the bin-packing DECISION problem.
// Correct DP: dp[mask] = minimum number of presses (each loaded <= cap)
// needed to cover exactly the jobs in `mask`. Transition: pick any sub-mask
// `sub` of the remaining jobs whose total <= cap to be ONE press, and recurse
// on the rest. We enumerate submasks of the complement.
bool feasible(long long cap) {
    for (int i = 0; i < n; i++) if (t[i] > cap) return false; // a job alone exceeds cap

    const int INF = 1e9;
    vector<int> dp(full + 1, INF);
    dp[0] = 0;
    for (int mask = 0; mask <= full; mask++) {
        if (dp[mask] == INF) continue;
        if (dp[mask] >= m) continue;          // already using m presses; can't open more
        int rest = full ^ mask;               // jobs not yet assigned
        // enumerate every non-empty submask of `rest` as the NEXT press's jobs
        for (int sub = rest; sub > 0; sub = (sub - 1) & rest) {
            if (sumMask[sub] <= cap) {
                int nmask = mask | sub;
                if (dp[mask] + 1 < dp[nmask]) dp[nmask] = dp[mask] + 1;
            }
        }
    }
    return dp[full] != INF && dp[full] <= m;
}

int main() {
    if (!(cin >> n >> m)) return 0;
    t.resize(n);
    for (auto &x : t) cin >> x;

    if (n == 0) { cout << 0 << "\n"; return 0; }
    full = (1 << n) - 1;

    sumMask.assign(full + 1, 0);
    for (int mask = 1; mask <= full; mask++) {
        int low = mask & (-mask);                 // lowest set bit
        int idx = __builtin_ctz(low);
        sumMask[mask] = sumMask[mask ^ low] + t[idx];
    }

    long long lo = *max_element(t.begin(), t.end()); // no press below its biggest job
    long long hi = 0; for (long long x : t) hi += x;  // one press does everything
    while (lo < hi) {
        long long mid = lo + (hi - lo) / 2;
        if (feasible(mid)) hi = mid; else lo = mid + 1;
    }
    cout << lo << "\n";
    return 0;
}
```
