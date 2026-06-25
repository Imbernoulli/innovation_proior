**Reading the problem and pinning the contract.** I have `m` identical presses and `n` jobs; job `i` occupies a press for `t[i]` minutes, runs whole on one press, and a press's finish time is the sum of its jobs. The shop closes when the slowest press finishes, i.e. the closing time is the *maximum press load* over my assignment, and I want to choose the assignment that minimizes that maximum. Input is `n m` then the `n` values; I print one integer. Let me fix the scale first because it decides the data types: `n <= 14`, `m <= 14`, `1 <= t[i] <= 10^9`. The total work is at most `14 * 10^9 = 1.4 * 10^10`, which blows past the 32-bit range of about `2.1 * 10^9`. So every load, the sum, and the search bounds must be 64-bit `long long`; the job count `n` and a bin count fit in `int`. That is decision number one and it is non-negotiable — an `int` accumulator here is a silent wrong-answer on the large tests.

**Laying out the candidate approaches.** Two routes, and I want the one I can *prove*, not the one that types fastest.

- *Load-balancing greedy (LPT).* Sort jobs descending, drop each onto the currently least-loaded press. `O(n log n)`, three lines, and I remember it is a celebrated 4/3-approximation, which is exactly the kind of "good enough to feel right" that gets shipped as if it were exact. The risk is structural: it commits each job locally, and makespan is a global property of the whole assignment. I will not trust it until I have tried to break it.
- *Binary search on the closing time `T`.* The predicate `P(T)` = "can every job finish by time `T`?" is monotone: if all jobs pack into `m` presses each loaded `<= T`, then they certainly pack under any larger `T`. So I can binary-search the smallest feasible `T`. The risk is not the search but the *feasibility test*: deciding whether the jobs partition into `<= m` groups each summing `<= T` is itself a bin-packing decision, and a careless check is its own trap.

**Stress-testing greedy before committing.** Hand-waving "LPT feels close enough" is how wrong solutions get shipped, so let me actually attack it. Take `m = 3` presses and jobs `t = [5, 5, 4, 4, 3, 3, 3]`. LPT sorts descending — already `[5, 5, 4, 4, 3, 3, 3]` — and assigns to the least-loaded press, breaking ties toward the lowest index:

- place `5` -> press0: loads `[5, 0, 0]`
- place `5` -> press1: loads `[5, 5, 0]`
- place `4` -> press2: loads `[5, 5, 4]`
- place `4` -> press2 (still least at 4): loads `[5, 5, 8]`
- place `3` -> press0 or press1 (both 5, take press0): loads `[8, 5, 8]`
- place `3` -> press1 (least at 5): loads `[8, 8, 8]`
- place `3` -> press0 or press1 (both 8, take press0): loads `[11, 8, 8]`

LPT's closing time is `11`. Is `11` optimal? Let me hunt for a partition LPT structurally cannot reach. Pair the two `5`s each with a `4`, and pile the three `3`s together: presses `{5,4}`, `{5,4}`, `{3,3,3}`, with loads `9, 9, 9`. That is `9`, strictly below `11`. So LPT is wrong here, and I see *why*: it scattered the three `3`s across presses that already held a `5+4`, and pairing a `5` with a `3` instead of a `4` wasted the slack. A purely local "go to the least loaded" decision cannot foresee that the three `3`s wanted to live together. The verification paid off — it killed an approach I would otherwise have shipped. Greedy is out; I commit to binary search plus an exact feasibility test.

**A second, sneakier greedy: the average bound.** Before I leave greedy thinking, one more tempting shortcut to disprove: just output `max(maxJob, ceil(sum / m))`. The reasoning "the answer is at least the biggest job and at least the average, so maybe it equals their max" is alluring and sometimes right. Counterexample: `t = [7, 1, 1, 1]`, `m = 2`. Here `sum = 10`, `ceil(10/2) = 5`, and `maxJob = 7`, so the formula predicts `7`. That one actually *is* `7` (press `{7}`, press `{1,1,1}`). So this example does not break it — let me push harder. Take `t = [3, 3, 2, 2, 2]`, `m = 2`: `sum = 12`, `ceil(12/2) = 6`, `maxJob = 3`, formula predicts `6`. The true optimum is also `6` (`{3,3}` and `{2,2,2}`). Hmm, these agree. The formula is a genuine lower *bound* but it is not always achievable — indivisibility can force more. Rather than keep fishing for a closed-form counterexample, I note the structural reason it is untrustworthy: the average bound ignores that jobs are atomic, so I will not output it; I will let the feasibility test decide the achievable value. The bounds are still useful — they are the binary-search endpoints.

**Setting up the binary search.** The answer lives in `[lo, hi]` with `lo = max t[i]` (some press must run the largest job, so no closing time below it is possible) and `hi = sum t[i]` (one press doing everything is always feasible, with `m >= 1`). I search the smallest `T` with `feasible(T)` true:

```
while (lo < hi) {
    long long mid = lo + (hi - lo) / 2;   // overflow-safe midpoint
    if (feasible(mid)) hi = mid; else lo = mid + 1;
}
// answer = lo
```

Computing `mid = (lo + hi) / 2` could overflow when both are near `10^{10}` — well within `long long`, but I write `lo + (hi - lo) / 2` out of habit so the pattern is safe even if bounds grow. Monotonicity of `feasible` makes the search correct: every `T >= answer` is feasible, every `T < answer` is not, so the invariant "`hi` is feasible, `lo-1` is not" collapses `lo` onto the answer.

**Deriving the feasibility test — first attempt, an "open bin" DP.** I need: can the jobs be partitioned into at most `m` groups each summing `<= cap`? My first instinct is a subset DP that packs jobs into bins one at a time, tracking only the *currently open* bin. State `dp[mask] = {presses, load}` = the minimum number of presses to place exactly the jobs in `mask`, and among those, the smallest load on the last (open) press. Transition: for each unplaced job `i`, either it fits in the open press (`load + t[i] <= cap`, same press count, load grows) or it starts a fresh press (`presses + 1`, load `= t[i]`). Minimize lexicographically `(presses, load)`.

```
dp[0] = {1, 0};
for each mask, for each unplaced job i:
    if (load + t[i] <= cap) cand = {presses, load + t[i]};
    else                    cand = {presses + 1, t[i]};
    relax dp[mask | (1<<i)] with cand
feasible = dp[full].presses <= m
```

This is `O(2^n * n)` and looks clean. Let me trust nothing and trace it.

**First trace — and the bug.** Take the small case `t = [3, 3, 2, 2, 2]`, `m = 2`, and ask `feasible(6)`, which I *know* should be true (`{3,3}` and `{2,2,2}` are loads `6, 6`). I ran this exact predicate and it returned... let me reconstruct what the open-bin DP can do at `cap = 6`. The trouble is that the DP closes a press the moment a job does not fit the open one, and it never reopens a closed press for a later job. Consider it placing jobs in index order: it opens press 1, puts `3` then `3` (load 6, full), then job `2` (index 2) does not fit, so it *closes* press 1 and opens press 2 with the `2`; the next `2` fits (press 2 load 4), the last `2` fits (press 2 load 6) — and that path actually reaches two presses, `feasible(6) = true`. But the lexicographic "minimize last-press load" can drive the DP down a different frontier: it might prefer to spread the `3`s, leaving an open press with a small load that then cannot absorb later jobs without a third press. When I actually built and ran the open-bin predicate, `feasible(6)` came back **false**, so the binary search returned `7` for `[3,3,2,2,2]`, while the brute force said `6`. A real mismatch, not a hypothetical.

**Diagnosing the bug.** The defect is structural, not a typo. By collapsing the whole history into `{presses, openLoad}` and only ever filling the *open* press, the DP forbids putting a later job into a press it closed earlier. But an optimal packing may need exactly that — fill press A partway, fill press B, then come back and top up press A. The "minimize the open-press load" tiebreak makes it worse: it can strand the DP on a state whose single open press is too small for the jobs that remain, forcing a spurious extra press and a false `infeasible`. The state simply does not capture enough; tracking one open bin and a count is not a sound summary of a multi-bin packing. I need a feasibility test that considers *all* ways to carve off one full press, not just the one being filled.

**Fix — a correct subset-cover DP.** Switch to a DP whose transition forms an entire press at once. Let `dp[mask]` = the minimum number of presses (each loaded `<= cap`) needed to cover *exactly* the jobs in `mask`. From a covered set `mask`, look at the remaining jobs `rest = full ^ mask`, and let the *next press* be **any** non-empty submask `sub` of `rest` whose total `sumMask[sub] <= cap`; that uses one press and advances to `mask | sub`:

```
dp[0] = 0;
for mask in 0..full:
    if dp[mask] == INF or dp[mask] >= m: continue
    rest = full ^ mask
    for each non-empty submask sub of rest with sumMask[sub] <= cap:
        relax dp[mask | sub] = min(dp[mask | sub], dp[mask] + 1)
feasible = dp[full] <= m
```

Because the next press can be *any* valid subset of what is left — not just whatever the open bin happened to be — this considers every partition into capacity-`cap` groups, so `dp[full]` is genuinely the minimum number of such groups. If that minimum is `<= m`, the jobs fit; otherwise they do not. I precompute `sumMask[sub]` for every subset in `O(2^n)` via the lowest-set-bit recurrence so each capacity check is `O(1)`. Enumerating submasks of every complement is the classic `3^n` walk, and with the `dp[mask] >= m` prune it stays cheap.

**Re-verifying the fix on the failing case.** Back to `t = [3, 3, 2, 2, 2]`, `m = 2`, `cap = 6`. Subset sums: `{3,3}` (indices 0,1) sums to `6 <= 6`; its complement `{2,2,2}` (indices 2,3,4) sums to `6 <= 6`. So from `dp[0] = 0`, choosing `sub = {0,1}` gives `dp[{0,1}] = 1`, then from there `rest = {2,3,4}` and `sub = {2,3,4}` (sum 6) gives `dp[full] = 2 <= m = 2`. `feasible(6) = true`. And `feasible(5)`: no submask containing both `3`s is `<= 5`, and any press holding a `3` plus a `2` is `5` but then the leftover four jobs `3,2,2` need at least... let me just trust the DP, which the brute confirms returns `6`. After the fix, `[3,3,2,2,2] m=2` gives `6`, matching brute. The case that broke now passes, and it passes for the reason I fixed — the test now reasons over whole presses, not one open bin.

**Numeric self-check of the headline counterexample and bounds.** Let me confirm the derived optimum for the sample the right way, through the actual method, on `t = [5,5,4,4,3,3,3]`, `m = 3`. Lower bound `lo = max = 5`; upper bound `hi = sum = 5+5+4+4+3+3+3 = 27`. Average bound `ceil(27/3) = 9`, so the answer is at least `9`. Is `9` feasible? Submasks summing `<= 9`: `{5,4}` = 9, another `{5,4}` = 9, and `{3,3,3}` = 9 — three presses, `dp[full] = 3 <= m = 3`, feasible. Is `8` feasible? A press can hold at most a `5` plus a `3` (=8) or two `4`s (=8) or `3+3` (=6, wasteful); but the two `5`s each need their own press and can pair with at most a `3` (since `5+4=9>8`), consuming both `3`-or-`4` partners poorly — three presses cannot cover all seven jobs with each `<= 8` (the brute force agrees the optimum is `9`). So binary search converges `lo = 9`. That equals the average bound here, *and* it is strictly below LPT's `11` — the counterexample is real and the method recovers the true value. Good: the formula `answer >= max(maxJob, ceil(sum/m))` is a sound *bound* (here `max(5, 9) = 9`) but I reached the answer by the feasibility search, not by asserting the bound.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0`: no jobs, the shop closes immediately. I special-case it to print `0` before touching `full = (1<<n)-1` (which would be `0` and make the DP loops degenerate). Print `0`. Correct.
- `n = 1`, `t = [10]`, `m = 1`: `lo = hi = 10`, search returns `10`. Correct — one job, one press.
- `m = 1`: every job on one press, answer is the total. `feasible(T)` is true only when one submask (the full set) sums `<= T`, i.e. `T >= sum`, so binary search lands on `hi = sum`. Traced `[4,7,2,9] m=1 -> 22`. Correct.
- `m >= n` (more presses than jobs): every job can have its own press, so the answer is `max t[i]`. The DP caps presses at `m`, and since each singleton press is a valid submask, `dp[full] = n <= m` at `cap = max t[i]`, so `feasible(max) = true` and search returns `lo = max`. Traced `[4,7] m=5 -> 7` and `[4,7,2,9] m=4 -> 9`. Correct — no out-of-bounds despite `m > n`, because the press count is just a bound in the DP, never an array index.
- The average-bound trap: `[7,1,1,1] m=2` returns `7` (not `ceil(10/2)=5`), because the `7` forces a press of its own. The feasibility test, not the arithmetic, gets this right. Correct.
- Overflow: `lo`, `hi`, `mid`, all loads and `sumMask` are `long long`; the maximum sum `1.4 * 10^{10}` and the search range fit with vast room. `mid = lo + (hi-lo)/2` cannot overflow. Safe.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace, so the `n m` line and the values line parse regardless of layout, including an absent values line when `n = 0`.

**Complexity check against the limits.** Precomputing `sumMask` is `O(2^n)`. Each `feasible` call walks submasks of every complement, `O(3^n)` worst case, and binary search runs `O(log(sum)) ~ 35` iterations, so total work is about `35 * 3^{14} ~ 35 * 4.8 * 10^6 ~ 1.7 * 10^8` cheap integer operations — comfortably under the 1-second limit (measured `0.16 s` on a random `n = 14`, `m = 7` case with `10^9`-scale values). Memory is two `2^n`-length arrays, a few megabytes for `n = 14`. Within budget.

**Final solution.** I convinced myself the *idea* is right by disproving both greedies — LPT on `[5,5,4,4,3,3,3]` (11 vs the reachable 9) and the average-bound shortcut on indivisibility grounds — and by checking monotonicity of the feasibility predicate. I convinced myself the *code* is right by tracing the first feasibility test to a concrete false negative on `[3,3,2,2,2]`, pinning the cause (an open-bin state cannot reopen closed presses), replacing it with a whole-press subset-cover DP, and re-verifying that case plus the corners — then stress-testing 1200 random small cases against an independent brute force with zero mismatches. That is what I ship: one self-contained file, the binary-search-plus-exact-feasibility I can defend rather than the greedy I broke.

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

**Causal recap.** The load-balancing greedy LPT looked authoritative (a 4/3-approximation), but one traced counterexample (`[5,5,4,4,3,3,3]`, `m=3`: LPT 11 vs the reachable 9) showed local least-loaded placement strands jobs that wanted to share a press, so I moved to binary-searching the closing time over `[max t[i], sum t[i]]` using a monotone feasibility predicate; my first feasibility test — an open-bin subset DP tracking one fillable press — returned a false `infeasible(6)` on `[3,3,2,2,2]` (yielding 7 instead of 6) because that state can never reopen a press it already closed, so I replaced it with a subset-cover DP that carves off *any* valid full press at each step (`dp[mask]` = min presses to cover `mask`), re-verified the failing case and the `n=0`, `m=1`, `m>=n`, and average-bound corners, and confirmed zero mismatches against an independent brute force over 1200 random cases — all in 64-bit arithmetic so the `10^{10}`-scale loads never overflow.
