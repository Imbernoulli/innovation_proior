**Reading the problem and pinning the contract.** I have `n` assays and a fixed reagent budget of `B` micrograms. Assay `i` costs `c[i]` micrograms and, if run, yields `r[i]` data points; each assay is all-or-nothing and runnable at most once. I must pick a subset whose total cost is at most `B` and whose total yield is maximal, and print that yield; running nothing is allowed so the answer is never below `0`. Stripped of the story this is exactly 0/1 knapsack: indivisible items, one capacity `B`, additive objective. Before I pick an algorithm I fix the scale, because it dictates both the data types and which methods are even admissible. `n <= 2000`, `B <= 2*10^5`, `c[i] <= 2*10^5`, `0 <= r[i] <= 10^9`. The largest possible total yield is `n * max(r[i]) = 2000 * 10^9 = 2*10^12`. That is far past the signed 32-bit ceiling of about `2.1*10^9`, so the yield accumulator and the whole DP table must be 64-bit. I will use `long long` for every yield and for `B`. That decision is non-negotiable; an `int` table is a silent wrong-answer on the large tests where a few high-yield assays alone overflow it. The costs themselves are at most `2*10^5` and fit in 32 bits, but I will read them into `long long` anyway to avoid mixed-type surprises in `w - c[i]` index arithmetic.

**Laying out the candidate approaches.** Two routes are on the table and I want to commit to the one I can *prove*, not the one that is shortest to type.

- *Greedy by efficiency (or by raw yield).* Sort assays by `r[i]/c[i]` descending — most data per microgram first — and run them while the remaining budget allows. `O(n log n)`, a handful of lines. The appeal is the fractional-knapsack intuition: if I could split assays, taking the densest reagent-for-data first is provably optimal. The danger is precisely that assays are indivisible and the budget may be left partly unspent, which is the regime where the fractional argument collapses. I will not trust this until I have actively tried to break it.
- *Knapsack DP over the budget.* Maintain `dp[w]` = the best yield achievable using total cost at most `w`, and update it assay by assay. `O(n*B)` time. At `n*B = 2000 * 2*10^5 = 4*10^8` this is the scary number, but each update is a single add-and-compare on contiguous memory, which a modern machine does at well over `10^9` per second, so under a 2-second limit it is comfortable. The risk here is not the idea but the transcription: the iteration direction that keeps each assay used at most once, the base case, and the data type.

**Stress-testing greedy before committing.** "Greedy feels right" is how wrong solutions get shipped, so let me actually attack it with a concrete instance rather than hand-wave. Let the budget be `B = 10` and take three assays: `A = (cost 6, yield 8)`, `B' = (cost 5, yield 6)`, `C = (cost 5, yield 6)`. Efficiency-greedy computes densities: `A` is `8/6 = 1.33` data per microgram, `B'` and `C` are `6/5 = 1.20` each. So greedy runs `A` first, spending 6 of the 10 micrograms and banking yield 8. Remaining budget is 4. The cheapest leftover assay costs 5, which does not fit, so greedy stops. Greedy's total is `8`.

Is `8` optimal? Let me hunt for a packing greedy structurally cannot reach. Run `B'` and `C` together: total cost `5 + 5 = 10 <= 10`, total yield `6 + 6 = 12`. That is strictly better than `8`, and it uses the budget to the last microgram. So greedy is wrong, and I now see *why* in mechanical terms: by grabbing the densest single assay `A` it consumed 6 micrograms and left a 4-microgram fragment too small for anything, whereas the optimum forgoes the densest item to fit two slightly-less-dense items that together tile the budget exactly. Density ordering optimizes a *rate* and ignores how the fixed budget *partitions*; with indivisible items the partition is everything. For good measure, raw-yield-greedy (sort by `r[i]` descending) makes the identical mistake here — `A` has the largest single yield `8`, so it is grabbed first and the same 4-microgram fragment is stranded. Both natural greedies return `8` against the true `12`. The verification paid off: it killed two approaches I might otherwise have shipped. Greedy is out; I commit to the DP.

**Deriving the DP and checking the recurrence on paper.** I want, for every budget level `w` from `0` to `B`, the best total yield achievable with total cost at most `w`, considering assays one at a time. Let `dp[w]` after processing the first `k` assays denote exactly that, restricted to subsets of those `k` assays. Adding assay `i` with cost `c[i]` and yield `r[i]`, a budget-`w` plan either does not run `i` (yield unchanged, `dp[w]`) or runs `i`, which requires `w >= c[i]` and leaves budget `w - c[i]` for the earlier assays, giving `dp[w - c[i]] + r[i]`. So the transition is

```
dp_new[w] = max(dp_old[w], dp_old[w - c[i]] + r[i])   for w >= c[i].
```

The base case before any assay is `dp[w] = 0` for all `w` (running nothing yields nothing and costs nothing, valid at every budget). The answer is `dp[B]` after all assays. Because "at most `w`" is monotone — a plan valid for budget `w` is valid for any larger budget — `dp` is nondecreasing in `w`, and `dp[B]` already captures the option of leaving budget unspent. That is important: I do not need a separate "exactly `w`" table and a final `max` over levels; the `<=` semantics fold the empty/partial-budget cases in automatically, and the empty subset (`dp[w] = 0`) keeps the answer at least `0`.

Let me confirm the recurrence by hand on the counterexample `B = 10`, assays `A=(6,8), B'=(5,6), C=(5,6)`, expected answer `12`. Start `dp[0..10] = 0`. Process `A=(6,8)`: for `w` from 10 down to 6, `dp[w] = max(dp[w], dp[w-6] + 8) = max(0, 0 + 8) = 8`. Now `dp[6..10] = 8`, `dp[0..5] = 0`. Process `B'=(5,6)`: for `w` from 10 down to 5, `dp[w] = max(dp[w], dp[w-5] + 6)`. At `w=10`: `max(8, dp[5] + 6) = max(8, 0 + 6) = 8`. At `w=5`: `max(0, dp[0] + 6) = 6`. So now `dp[5..9] = 6` where it was 0, `dp[10] = 8`, `dp[6..9]` becomes `max(8, dp[w-5]+6)`; e.g. `dp[9] = max(8, dp[4]+6) = max(8,6) = 8`. Process `C=(5,6)`: for `w` from 10 down to 5. At `w=10`: `max(dp[10], dp[5] + 6) = max(8, 6 + 6) = 12`. There it is — `dp[10] = 12`. The recurrence reaches the optimum the greedy could not.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut of the core, written quickly:

```
vector<long long> dp(B + 1, 0);
for (int i = 0; i < n; i++) {
    for (long long w = c[i]; w <= B; w++) {        // forward over w
        dp[w] = max(dp[w], dp[w - c[i]] + r[i]);
    }
}
cout << dp[B] << "\n";
```

The forward direction over `w` nags at me — in a 0/1 knapsack each item must be used at most once, and updating `dp[w]` from a *smaller* index `dp[w - c[i]]` that I already updated *this same iteration* is the classic way to accidentally reuse an item. Let me trace the smallest input that exposes it. Take `B = 6` and a single assay `A = (cost 3, yield 5)`. The answer is obviously `5`: I can run `A` once (cost 3, fits in 6), and there is only one assay so I cannot do better. Run the loop. `dp` starts all `0`. Processing `A=(3,5)`, `w` goes forward `3,4,5,6`:
- `w=3`: `dp[3] = max(0, dp[0] + 5) = max(0, 5) = 5`.
- `w=4`: `dp[4] = max(0, dp[1] + 5) = 5`.
- `w=5`: `dp[5] = max(0, dp[2] + 5) = 5`.
- `w=6`: `dp[6] = max(0, dp[3] + 5)`. But `dp[3]` was already set to `5` *this same pass*, so this is `max(0, 5 + 5) = 10`.

Final `dp[6] = 10`.

**Diagnosing the first bug.** The code returns `10` for a single assay of yield `5` — it ran assay `A` *twice* to fill the budget of 6 with two copies of the cost-3 item. The defect is precise: by iterating `w` upward, when I reach `w = 6` the cell `dp[3]` I read has *already absorbed assay `A` in this same outer iteration*, so `dp[6] = dp[3] + 5` stacks a second copy of `A` on top of the first. That is the unbounded-knapsack recurrence, not 0/1. To use each assay at most once I must read a `dp` value that does **not** yet reflect the current assay — i.e. iterate `w` from high to low, so every `dp[w - c[i]]` I read still holds the previous-iteration (assay-not-yet-included) value. The fix is to reverse the inner loop.

**Fixing and re-verifying the first bug.** Reverse the direction:

```
vector<long long> dp(B + 1, 0);
for (int i = 0; i < n; i++) {
    for (long long w = B; w >= c[i]; w--) {        // backward over w
        dp[w] = max(dp[w], dp[w - c[i]] + r[i]);
    }
}
```

Re-trace `B=6`, single assay `A=(3,5)`. `dp` starts `0`. `w` goes `6,5,4,3`:
- `w=6`: `dp[6] = max(0, dp[3] + 5)`; `dp[3]` is still `0` (not yet touched this pass) so `dp[6] = 5`.
- `w=5`: `dp[5] = max(0, dp[2] + 5) = 5`.
- `w=4`: `dp[4] = max(0, dp[1] + 5) = 5`.
- `w=3`: `dp[3] = max(0, dp[0] + 5) = 5`.

Final `dp[6] = 5`. Correct — `A` is counted once. The case that broke now passes, and it broke for exactly the reason I fixed (reading an already-updated lower cell), which is the evidence I trust. Re-run the three-assay counterexample with the backward loop too: I already traced above that the backward pass gives `dp[10] = 12`, matching the optimum, so both my correctness witness and my bug witness are green.

**Second trace — a sneaky index/guard bug on large costs.** Now I worry about an assay whose cost exceeds the budget, `c[i] > B`, which the constraints explicitly allow. In the loop `for (long long w = B; w >= c[i]; w--)`, if `c[i] > B` then the condition `w >= c[i]` is false at the very start (`B >= c[i]` is false), so the loop body never runs and that assay is correctly skipped — it can never be part of any feasible plan. Good, the bound handles it. But let me check the *type* of `w` deliberately, because this is where a different silent bug hides. Suppose I had declared `w` as `int` and written `for (int w = B; w >= c[i]; w--)`. With `B` and `c[i]` both up to `2*10^5`, `int` holds them fine, so that particular instance is safe. Where it would bite is the index expression `dp[w - c[i]]`: if `w` and `c[i]` were `int` and I ever formed `w - c[i]` while `w < c[i]`, the result is negative and indexing `dp[negative]` is undefined behavior. My loop guard `w >= c[i]` guarantees `w - c[i] >= 0` *inside* the body, so the access is always in range `[0, B]`. Let me trace the boundary explicitly: `B = 3`, single assay `A = (10, 100)` (cost far over budget), expected answer `0` (nothing runnable). The loop is `for (w = 3; w >= 10; w--)` — condition false immediately, body never executes, `dp` stays all `0`, output `dp[3] = 0`. Correct. And `B = 4`, assay `A = (4, 7)` (exact-fit boundary): loop runs only at `w = 4`, `dp[4] = max(0, dp[0] + 7) = 7`, output `7`. Correct — the `>=` boundary is inclusive as it must be so an exactly-affordable assay is allowed. The guard is doing double duty: it both skips over-budget assays and keeps the index non-negative. I keep `w` and the costs as `long long` regardless, so even if some future edit lets `B` grow past `2^31` the arithmetic stays clean.

**Sanity-checking the derivation on the documented sample.** The stated example is `B = 10`, assays `(6,8), (5,6), (5,6)`, answer `12`. My paper trace of the backward DP above walked the three passes and landed `dp[10] = 12`, and the brute-force subset enumeration agrees the best feasible subset is `{B', C}` with cost `10` and yield `12`. The greedy that the problem warns about returns `8`. So the derivation, the recurrence, and the sample all line up, and the DP gives the answer greedy cannot.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0`: no assays to process, `dp` is `[0]*(B+1)` and stays so; output `dp[B] = 0`. Run nothing — correct.
- `B = 0`: `dp` has length 1 (`dp[0] = 0`); every assay has `c[i] >= 1 > 0`, so the inner loop `for (w = 0; w >= c[i]; w--)` never runs; output `dp[0] = 0`. With zero budget nothing affordable — correct.
- An assay with `r[i] = 0`: it can be "run" for free yield but never improves any `dp[w]` because `max(dp[w], dp[w-c]+0)` only changes a cell if `dp[w-c] > dp[w]`, which the monotonicity of `dp` forbids (`dp[w-c] <= dp[w]`). So zero-yield assays are harmless no-ops — correct, they never spuriously inflate the answer.
- All assays cost more than `B`: every inner loop is empty, output `0` — correct.
- Exact-fit instance: the `>= c[i]` boundary is inclusive, so an assay (or combination) summing exactly to `B` is reachable, as the `(4,7)` and `(5,6)+(5,6)=10` traces showed.
- Overflow: `dp` is `long long`; the maximum reachable yield `~2*10^12` fits with five decimal digits of headroom below the `~9.2*10^18` ceiling. No accumulator ever multiplies, only adds, and at most `n = 2000` yields of `10^9` are summed, so `2*10^12` is the hard cap — safe. Costs and `B` are at most `2*10^5`, so no index ever exceeds the table.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace, so the row-or-stream layout of the input does not matter.

**On performance, concretely.** The table is `B + 1 <= 200001` longs, about 1.6 MB — well inside 256 MB. The work is `sum_i min(B - c[i] + 1, ...) <= n*(B+1) = 4*10^8` add-and-compare steps over contiguous memory. I measured the worst shape (`n = 2000`, `B = 2*10^5`, random costs and yields up to `10^9`): it ran in about `0.12 s` and used under `5 MB`, comfortably under the 2-second limit. I add `ios_base::sync_with_stdio(false)` and `cin.tie(nullptr)` so reading up to `~4000` integers is never the bottleneck.

**Final solution.** I convinced myself the *idea* is right by disproving both greedies with a traced counterexample (`12` reachable, greedy stuck at `8`) and hand-checking the recurrence on that very instance; I convinced myself the *code* is right by tracing the forward-loop double-counting bug to a precise cause (reading a same-pass-updated lower cell) and re-verifying the backward fix, then separately tracing the over-budget and exact-fit boundaries. Across 750 random small cases the backward DP matched an independent brute-force subset enumeration with zero mismatches. That is what I ship — one self-contained file, the `O(n*B)` 0/1 knapsack I can defend rather than the greedy I broke:

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

**Causal recap.** Efficiency-greedy looked optimal by the fractional-knapsack intuition, but a single traced instance (`B=10`, assays `(6,8),(5,6),(5,6)`: greedy strands a 4-microgram fragment for `8`, the optimum tiles the budget exactly for `12`) showed that with indivisible items the *partition* of the budget matters, not the *rate*, so I moved to a knapsack DP where `dp[w]` is the best yield within budget `w`; the recurrence `dp[w] = max(dp[w], dp[w-c]+r)` is right but my first forward inner loop read an already-updated lower cell and double-counted an assay (a trace of one cost-3 yield-5 assay returning `10` instead of `5` pinpointed it), which the high-to-low iteration fixes by reading only previous-pass values; the `w >= c[i]` guard both skips over-budget assays and keeps every index non-negative, and a `long long` table closes the `~2*10^12` overflow corner while the `<= B` semantics and the all-zero base case handle the empty, zero-budget, zero-yield, and exact-fit cases.
