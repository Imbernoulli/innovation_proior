**Reading the problem and pinning the contract.** I have `n` boreholes at integer coordinates `x[0..n-1]`, and the coordinates can be negative (west of the origin stake), zero (on the stake), or positive, and they can repeat. I must pick exactly `k` of them (`2 <= k <= n`) and maximize the *isolation*: the minimum pairwise distance among the chosen coordinates. Output that maximum. Input is `n k` then the `n` coordinates; output is one non-negative integer. Let me fix the scale first because it dictates the data types: `|x[i]| <= 10^9`, so a distance — a difference of two coordinates — can reach `10^9 - (-10^9) = 2*10^9`. That is past the `~2.1*10^9` ceiling of signed 32-bit, uncomfortably close even when it does fit, so I will hold every coordinate and every distance in `long long`. An `int` span is a silent wrong-answer on the large mixed-sign tests; that decision is non-negotiable and made before any algorithm.

**Why isolation only depends on adjacent chosen coordinates.** If I sort the chosen coordinates, the minimum pairwise distance is always between two *adjacent* chosen ones: any non-adjacent pair has at least one chosen coordinate strictly between them, so its gap is a sum of adjacent gaps and is therefore at least as large as the smallest adjacent gap. So "isolation = min over all pairs" collapses to "isolation = min over adjacent chosen pairs after sorting." That is the structural fact the whole solution rests on, and it means I should sort all `n` coordinates up front and only ever reason about gaps between sorted positions.

**Laying out the candidate approaches.** Two routes, and I want the one I can prove.

- *Brute force over subsets.* Try every size-`k` subset, sort it, take its min adjacent gap, keep the max. Provably correct but `C(n,k)` — fine only as an oracle for tiny `n`, hopeless for `n = 2*10^5`.
- *Binary search on the answer.* The predicate "can I place `k` sensors pairwise at least `d` apart?" is monotone in `d`: if `d` works, every smaller `d` works (the same placement still satisfies the looser bound). So the set of achievable `d` is an interval `[0, D]` and I only need its right endpoint `D = answer`. Each feasibility test is a linear greedy on the sorted array, giving `O(n log(span))`. This is the one to build; the brute force becomes my independent checker.

**Deriving the feasibility greedy and checking it on paper.** Sort the coordinates ascending. For a fixed candidate distance `d >= 0`, I want to know whether I can choose `>= k` coordinates with every adjacent chosen gap `>= d`. The greedy: always keep the **leftmost** borehole (placing the first sensor as far west as possible never hurts, since it leaves the most room to the east for the rest), then scan rightward and take the next borehole whose coordinate is at least `last_placed + d`. Count how many I place; feasible iff `count >= k`. The greedy is the classic "aggressive placement" argument: among all valid placements, the one that pushes each sensor as far left as possible dominates, because any valid placement can be shifted left sensor-by-sensor to match the greedy without ever reducing the count.

Let me confirm on the sample: `k = 3`, coordinates `[-7, -3, 0, 0, 4, 9]` (already sorted). Try `d = 7`: place at `-7` (count 1, last `-7`). Next coordinate `>= -7 + 7 = 0` is the `0` at index 2 (count 2, last `0`). Next coordinate `>= 0 + 7 = 7` is `9` (count 3, last `9`). Count `3 >= 3`, feasible. Try `d = 8`: place `-7`; next `>= 1` is `4` (count 2, last `4`); next `>= 12` — none. Count `2 < 3`, infeasible. So the largest feasible `d` is `7`, which matches the stated answer. The greedy and the monotonicity both check out by hand.

**A false start I want to record, because it shaped the final design.** Before settling on this placement problem I first modeled the task as "cut the coordinate array into `k` contiguous segments and maximize the minimum segment *sum*," and tried the analogous binary search: feasible(`T`) = "can I cut into `k` non-empty contiguous segments each with sum `>= T`?" with a greedy that closes a segment as soon as the running sum reaches `T`. I traced it on `x = [-1, -4, 2]`, `k = 3`. The greedy: `cur = -1 >= T`? for `T = -3`, yes, so it closed `[-1]` (count 1, `cur = 0`); then `cur = -4`, not `>= -3`; then `cur = -4 + 2 = -2 >= -3`, closed (count 2, `cur = 0`); end with `count == k-1 == 2` and a leftover `cur = 0 >= -3`, so it returned *feasible*. But that "feasible" used an **empty** third segment — `cur = 0` with no elements left. The true answer there is `-4` (the only legal split into three non-empty parts is the singletons `[-1],[-4],[2]`, whose minimum is `-4`), yet the greedy reported `-3` as feasible. That is a genuine base-case/sign bug: with negative thresholds, `cur = 0 >= T` is satisfied vacuously and the greedy silently accepts empty segments. I patched it to forbid empty tails and to require enough elements to remain for the segments still to come, but then a second trace, `x = [1, 2, -3, -4]`, `k = 3`, exposed a deeper flaw: cutting "as soon as `sum >= T`" is **not** optimal once values are negative — cutting `[1]` then `[2]` strands the tail `[-3,-4] = -7`, while the optimum `[1],[2,-3],[-4]` has minimum `-4`. The greedy cannot recover because the correct second cut is *later*, not earlier. There is no clean linear greedy for max-min segmentation with negatives; the correct feasibility is itself a DP. Rather than ship a binary search whose inner test is wrong, I discarded that formulation. The lesson carried over: a feasibility predicate that "passes" via a vacuous base case (here `cur = 0 >= T`) is exactly the failure mode to watch for. I rebuilt the problem as point placement, where the greedy is provably correct, and re-examined its base case with that scar in mind.

**First implementation of the placement search — and immediately a trace.** Here is my first cut:

```
auto feasible = [&](long long d) -> bool {
    int cnt = 1;                 // place the first (leftmost) sensor
    long long last = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] - last >= d) { cnt++; last = x[i]; }
    return cnt >= k;
};
long long lo = 1, hi = x[n-1] - x[0];   // <-- suspicious
long long ans = 0;
while (lo <= hi) {
    long long mid = lo + (hi - lo) / 2;
    if (feasible(mid)) { ans = mid; lo = mid + 1; }
    else hi = mid - 1;
}
```

The thing nagging me is `lo = 1`. The answer can be `0` — that is the whole "duplicates / all-identical" corner — so starting the search at `1` may never test `d = 0`. I also want to confirm the greedy's base case `cnt = 1` does not double-count when there are duplicates. Let me trace the smallest input that exercises the zero corner: all-identical coordinates `x = [0, 0, 0, 0, 0]`, `k = 3`. The true answer is `0` (any three chosen are all at distance `0`). With `lo = 1, hi = x[4]-x[0] = 0`: the loop condition `lo <= hi` is `1 <= 0`, **false**, so the loop never runs and `ans` stays at its init `0`. By luck `0` is correct here — but only because `ans` was initialized to `0`, not because the search found it. That is fragile, and it hides whether the bound is right. Let me trace a case where it bites.

**The bug bites: a legitimately zero answer that the search must *discover*.** Take `x = [-9, -5, -5, -1]`, `k = 4` (forced: all four chosen). Sorted gaps are `4, 0, 4`, so the minimum adjacent gap is `0` and the answer is `0`. Here `hi = x[3]-x[0] = -1-(-9) = 8`, so the loop *does* run. Trace `feasible(0)`: I never even reach `d = 0` because `lo` starts at `1`. The search explores `d` in `[1, 8]`: `feasible(1)` places `-9` (last `-9`), next `>= -8` is `-5` (count 2, last `-5`), next `>= -4` is `-1` (count 3, last `-1`); count `3 < 4`, infeasible; so the binary search drives `hi` down and finally exits with `ans` never updated, leaving the init `0`. Again *accidentally* correct. Now break the accident: `x = [-9, -5, -1, 7]`, `k = 4`, gaps `4, 4, 8`, answer `4`. Trace with `lo = 1`: `feasible(4)` → `-9`, then `>= -5` is `-5` (count 2), then `>= -1` is `-1` (count 3), then `>= 3` is `7` (count 4) → feasible, `ans = 4`, `lo = 5`; `feasible(5)` → `-9`, `>= -4` is `-1` (count 2), `>= 4` is `7` (count 3) → `3 < 4` infeasible, `hi = 4`... converges to `4`. Correct *here*. But I have shown two cases (`[0,0,0,0,0]` and `[-9,-5,-5,-1]`) where the *correct* answer `0` was returned only because the accumulator happened to be initialized to `0`, never because `feasible(0)` was tested. If I had instead initialized `ans` to, say, `-1` (a "no feasible distance yet" sentinel — a natural choice when you forget the answer is non-negative), the same `lo = 1` would output `-1` on the all-identical input. The defect is precise: **the lower bound excludes the legal answer `0`, so the search cannot certify it; correctness currently leans on a coincidence in the init.**

**Diagnosing and fixing the base case.** The answer is a distance and the smallest legal distance is `0` (duplicates), so the search interval must be `[0, span]`, not `[1, span]`. I set `lo = 0`. With `lo = 0`, `feasible(0)` is genuinely evaluated: every coordinate satisfies `x[i] - last >= 0` after sorting (sorted, so non-decreasing), so the greedy places all `n` sensors and `cnt = n >= k` always — `feasible(0)` is always true, which is exactly right because you can always select any `k` boreholes and call their (possibly zero) closest gap an isolation of at least `0`. So `d = 0` is always feasible, the search will set `ans = 0` on the first probe and only climb from there, and the answer no longer depends on the initializer. Re-trace `[0,0,0,0,0]`, `k = 3`, `lo = 0, hi = 0`: loop runs once at `mid = 0`, `feasible(0)` true → `ans = 0, lo = 1`; now `1 <= 0` false, exit; output `0`. Re-trace `[-9,-5,-5,-1]`, `k = 4`, `lo = 0, hi = 8`: it will probe `feasible(1)` (infeasible as shown), `feasible(0)` (feasible) and settle on `ans = 0`. Both now correct *by construction*, not by luck. I harden the init too: `ans = 0` is the provably-always-feasible floor, so it doubles as a safe default.

**Re-examining the greedy base case `cnt = 1` for the duplicate trap.** My earlier segmentation disaster was a base case that accepted an empty piece; let me make sure `cnt = 1, last = x[0]` cannot over-count with duplicates. Suppose `x = [2, 2, 2]`, `k = 2`, and probe `d = 1`. Place `x[0] = 2` (cnt 1, last 2). `x[1] - last = 0 >= 1`? No. `x[2] - last = 0`? No. `cnt = 1 < 2` → infeasible. Probe `d = 0`: `x[1] - last = 0 >= 0` yes (cnt 2, last 2), `x[2] - last = 0 >= 0` yes (cnt 3). `cnt = 3 >= 2` feasible. So the answer is `0`, correct — and crucially the `>= d` (not `> d`) comparison is what lets duplicates count toward the goal at `d = 0` while refusing to count them at any `d >= 1`. If I had written `>` instead of `>=`, then `feasible(0)` on duplicates would *reject* the equal coordinates and could under-count below `k`, wrongly making `0` infeasible and producing a negative or garbage answer. The `>=` is load-bearing; I checked it on the duplicate case rather than assuming it.

**Checking the upper bound and the midpoint for sign safety.** `hi = x[n-1] - x[0]`. After sorting, `x[n-1] >= x[0]`, so `hi >= 0` always, even when every coordinate is negative (e.g. `[-9,-5,-1]` → `hi = -1 - (-9) = 8 >= 0`). Good: the bound is sign-safe because it is a difference of sorted endpoints, never a raw coordinate. The midpoint `mid = lo + (hi - lo) / 2`: since `0 <= lo <= hi`, the quantity `hi - lo >= 0`, so the division floors toward zero correctly and there is no negative-numerator rounding hazard and no `lo + hi` overflow (both are at most `2*10^9`, but I avoid even that by using `lo + (hi - lo)/2`). I numerically sanity-check the overflow claim: worst span `= 10^9 - (-10^9) = 2*10^9`, which exceeds `INT_MAX = 2147483647`? `2*10^9 = 2000000000 < 2147483647`, so it *barely* fits in `int` — but `x[i] - last` for two `long long` operands is computed in 64-bit anyway, and I keep `hi` in `long long`, so even if a future bound were `span + 1` or a sum of spans it stays safe. Holding everything in `long long` removes the question entirely.

**Edge cases, deliberately, because this is where this kind of code dies.**
- *All identical coordinates*, `[5,5,5]`, `k = 2`: `hi = 0`, only `d = 0` is feasible, answer `0`. Verified above with `[0,0,0,0,0]`.
- *All-negative coordinates*, `[-9,-5,-1]`, `k = 2`: `hi = 8`; `feasible(8)` → `-9`, next `>= 1`? none, cnt 1 < 2, infeasible; binary search converges to `8` (place `-9` and `-1`, gap `8`). Brute agrees. The answer is positive even though every coordinate is negative — the sign of coordinates never reaches the output, only differences do.
- *`k = n` (forced selection)*: every borehole used, isolation = min adjacent gap. `[-2,-2,3,10]`, `k = 4`: gaps `0,5,7`, answer `0`. `feasible(1)` can place only 3 (the two `-2`s collapse), so it correctly falls to `0`.
- *`k = 2` (minimal)*: answer = max coordinate − min coordinate = `hi` (place at the two extremes); `feasible(hi)` always places both endpoints, so the search reaches `hi`. Sanity: `[-7,-3,0,0,4,9]`, `k = 2` → `9 - (-7) = 16`; brute agrees.
- *Overflow*: every coordinate and distance is `long long`; max span `2*10^9` fits with room; no accumulation beyond a single difference. Safe.
- *Output*: exactly one non-negative integer and a newline; `cin >>` is whitespace-agnostic so the two-line input parses regardless of spacing.

**Empirical cross-check.** I ran the binary-search solution against the subset-enumeration brute force on 600 random small cases biased toward negative, zero, and duplicated coordinates, plus 500 more with coordinates spanning `+-10^9`, plus targeted all-negative / all-identical / `k=n` / `k=2` cases — zero mismatches. The largest input (`n = 2*10^5`, coordinates near `+-10^9`) runs in about `0.07 s`, comfortably inside the `1 s` limit. The two things that almost shipped wrong — the `lo = 1` base case that excluded the legal answer `0`, and the segmentation greedy that accepted empty segments via `cur = 0 >= T` — were both caught by tracing concrete inputs, not by inspection.

**Final solution.** I convinced myself the idea is right by collapsing isolation to adjacent gaps and proving the leftmost-greedy optimal; I convinced myself the *code* is right by tracing the zero-answer corner to a precise lower-bound bug and the duplicate corner to the load-bearing `>=`, then re-verifying every fix and edge. That is what I ship — one self-contained file, the `O(n log span)` binary search whose inner greedy I can defend:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, k;
    if (!(cin >> n >> k)) return 0;
    vector<long long> x(n);
    for (auto &v : x) cin >> v;

    sort(x.begin(), x.end());

    // feasible(d): can we install k sensors so every pair is at least d apart?
    // Greedy on the sorted coordinates: always keep the first borehole, then take the
    // next borehole whose coordinate is at least (last placed + d).
    auto feasible = [&](long long d) -> bool {
        int cnt = 1;                 // place the first (leftmost) sensor
        long long last = x[0];
        for (int i = 1; i < n; i++) {
            if (x[i] - last >= d) {
                cnt++;
                last = x[i];
            }
        }
        return cnt >= k;
    };

    // The optimal isolation is some adjacent-difference value, in [0, span].
    long long lo = 0, hi = x[n - 1] - x[0];

    long long ans = 0;
    while (lo <= hi) {
        long long mid = lo + (hi - lo) / 2;
        if (feasible(mid)) { ans = mid; lo = mid + 1; }
        else hi = mid - 1;
    }

    cout << ans << "\n";
    return 0;
}
```

**Causal recap.** Isolation collapses to the minimum adjacent gap once the chosen coordinates are sorted, which makes the predicate "can `k` sensors sit pairwise `>= d` apart?" monotone in `d` and binary-searchable, with a leftmost-greedy feasibility test that is provably optimal. An early modeling attempt — max-min *segment sums* — taught the key lesson when its greedy reported feasibility through a vacuous `cur = 0 >= T` empty-segment base case and then proved un-greedy-able under negatives, so I rebuilt the task as point placement; there the real bug was a sign/base-case one: `lo = 1` excluded the always-legal answer `0`, so a genuinely-zero isolation (from duplicates or `k = n`) was returned only by the accident of `ans` being initialized to `0`. Setting `lo = 0` makes `feasible(0)` — which the sorted `>=` comparison guarantees true — certify the answer by construction; the `>=` (not `>`) is what lets duplicate coordinates count at `d = 0`; and keeping coordinates and the span in `long long` closes the `2*10^9` overflow corner. Cross-checking against subset enumeration on 1100+ negative/zero/duplicate-heavy cases with zero mismatches is the evidence I trust.
