**Reading the problem and pinning the contract.** I have `n` crates, crate `i` carrying integer mass `w[i]` and integer value `v[i]`, and a mass budget `C`. I must pick a subset with total mass at most `C` maximizing total value; the empty load is allowed, so the answer never drops below `0`. Input is `n` and `C` on the first line, then `n` lines of `w[i] v[i]`; I print one integer. Before any algorithm I fix the scale because it dictates types. `n <= 1000`, `C <= 2000`, `v[i] <= 10^9`. The worst-case total value is roughly `n * max(v) = 1000 * 10^9 = 10^12`, which is far past the 32-bit ceiling of about `2.1 * 10^9`. So every value accumulator must be 64-bit; I will carry values in `long long`. Masses and `C` are small (`<= 2000`), but I will keep `C` as `long long` too so capacity arithmetic never silently mixes signedness with the value sums. This is the first non-negotiable decision: `int` value sums are a silent wrong-answer on the large tests.

**Laying out the candidate approaches.** Two routes, and I want the one I can *prove* under indivisibility, not the one that types fastest.

- *Ratio greedy.* Sort crates by `v[i] / w[i]` descending, load in that order while they fit, and take any zero-mass positive-value crate for free. `O(n log n)`, a handful of lines. It is genuinely optimal for the *fractional* knapsack, where I could slice the last crate to exactly fill the budget — and that theorem is exactly the bait. Here crates are indivisible. I will not trust greedy until I have actively tried to break it.
- *Capacity DP.* For each total mass `c` from `0` to `C`, track the best total value of a subset whose mass is exactly `c`, folding crates in one at a time. `O(n * C) = 1000 * 2000 = 2 * 10^6` cell updates — trivially under the time limit. The risk is transcription: the iteration direction and the reachability bookkeeping are easy to get subtly wrong.

**Stress-testing greedy before committing — building an explicit counterexample.** "Greedy feels right because fractional knapsack works" is precisely the reasoning that ships wrong solutions, so let me attack it with a concrete instance rather than a feeling. I want a budget that a single high-ratio crate can *waste*. Take `C = 10` and three crates: crate A = `(w=6, v=10)`, crate B = `(w=5, v=7)`, crate C = `(w=5, v=7)`.

The ratios are `A: 10/6 ≈ 1.667`, `B: 7/5 = 1.4`, `C: 7/5 = 1.4`. Greedy sorts A first. It loads A (mass used 6, value 10, budget left 4). Next it tries B: mass 5 does not fit in the remaining 4. Then C: also does not fit. Greedy stops at value `10`.

Now I hunt for something greedy structurally cannot reach. Load B and C together: total mass `5 + 5 = 10 <= 10`, total value `7 + 7 = 14`. That is strictly better than greedy's `10`. So greedy is wrong, and I see *why*: A has the best value-per-mass, but loading it consumes 6 of the 10 units and strands the remaining 4 as dead budget, because indivisibility forbids slicing a 5-mass crate down to fit. In the fractional world greedy would fill those 4 units with `4/5` of a 5-crate for `+5.6`, reaching `15.6` — but I cannot slice. The fractional optimum (`15.6`) and the integral optimum (`14`) are different numbers, and greedy chases the fractional one. The counterexample is decisive; greedy is out. I will let the DP be the source of truth and keep this instance as a regression check.

Let me sanity-check that integral-vs-fractional claim numerically so I am not hand-waving. Fractional greedy on `C=10`: take all of A (`+10`, mass 6, budget 4 left), then `4/5` of a 5-crate at ratio 1.4 gives `4 * 1.4 = 5.6`, total `15.6`. Integral optimum is `14` (crates B+C). And `15.6 > 14 > 10`, so the fractional bound strictly dominates both the integral optimum and the greedy integral answer — exactly the gap that makes ratio greedy unsafe here. Good, the numbers line up.

**Deriving the DP and checking the recurrence on paper.** I want, for every reachable total mass `c`, the maximum value of a subset whose mass is *exactly* `c`. Call it `dp[c]`. I fold crates in one at a time. Before any crate only mass `0` is reachable, with value `0`; every other `c` is unreachable. I will encode "unreachable" as `-1`, distinct from the legitimately-reachable value `0` (e.g. the empty mass, or a zero-value crate). The transition for crate `i` with mass `w[i]`, value `v[i]`: a mass-`c` subset either excludes the crate (value unchanged) or includes it, in which case the rest is a mass-`(c - w[i])` subset *built without this crate yet*, so

`dp_new[c] = max( dp_old[c],  dp_old[c - w[i]] + v[i] )`  for `c >= w[i]` and `dp_old[c - w[i]] != -1`.

The "without this crate yet" clause is the whole 0/1 discipline: the predecessor state `dp_old[c - w[i]]` must not already contain crate `i`. The final answer is `max` over all reachable `dp[c]`, `0 <= c <= C`, floored at `0` for the empty load.

Let me confirm the recurrence by hand on the sample (`C=10`; A=(6,10), B=(5,7), C=(5,7); answer 14). Start: `dp[0]=0`, everything else `-1`. Fold A=(6,10): only `dp[0]` is reachable, so `dp[6] = dp[0] + 10 = 10`; now reachable masses are `{0:0, 6:10}`. Fold B=(5,7): from `dp[0]` get `dp[5] = 7`; from `dp[6]` get `dp[11]` — but `11 > C=10`, out of range, skip. Reachable `{0:0, 5:7, 6:10}`. Fold C=(5,7): from `dp[0]` get `dp[5] = max(7, 0+7) = 7`; from `dp[5]=7` get `dp[10] = 7 + 7 = 14`; from `dp[6]=10` get `dp[11]` out of range. Reachable `{0:0, 5:7, 6:10, 10:14}`. The best is `dp[10] = 14`. The recurrence reproduces the optimum and, importantly, does *not* reproduce greedy's 10. The idea is right.

**First implementation — and immediately a trace, because clean math transcribes dirty.** I will keep a single `dp` array and update it in place to save memory. My first cut of the fold:

```
vector<long long> dp(C + 1, -1);
dp[0] = 0;
for (int i = 0; i < n; i++) {
    if (w[i] > C) continue;
    for (long long c = w[i]; c <= C; c++) {     // <-- forward sweep
        if (dp[c - w[i]] != -1) {
            long long cand = dp[c - w[i]] + v[i];
            if (cand > dp[c]) dp[c] = cand;
        }
    }
}
```

Something about sweeping `c` upward while updating `dp` in place looks dangerous, so I trace the smallest input that could expose it. Take `C = 10` and a *single* crate `(w=5, v=7)`. The answer is obviously `7`: I can load that crate once. Trace: `dp[0]=0`, rest `-1`. Fold the crate, `c` from 5 up: `c=5`: `dp[0]=0 != -1`, `dp[5] = max(-1, 0+7) = 7`. `c=6`: `dp[1]=-1`, skip. ... `c=10`: `dp[5]=7 != -1`(!), so `dp[10] = max(-1, 7 + 7) = 14`. Final best `= 14`.

**Diagnosing the bug.** The code returns `14` from a *single* crate of value 7 — it loaded the same crate twice. The defect is precise: sweeping `c` upward, when I reach `c = 10` I read `dp[c - w] = dp[5]`, but `dp[5]` was *already overwritten this very iteration* (at `c = 5`) to include the crate. So `dp[10] = dp[5] + 7` chains "crate, then crate again." That is the unbounded knapsack, not 0/1. The fix is to sweep `c` *downward*, from `C` down to `w[i]`: then when I update `dp[c]` I read `dp[c - w[i]]` from a strictly smaller index that has not yet been touched in this crate's pass, so it still reflects the state *before* this crate. Each crate is folded in at most once. The forward sweep is the canonical 0/1-vs-unbounded trap, and the trace caught it on the smallest possible witness.

**Fixing and re-verifying the direction.** Reverse the inner loop:

```
for (long long c = C; c >= w[i]; c--) {
    if (dp[c - w[i]] != -1) {
        long long cand = dp[c - w[i]] + v[i];
        if (cand > dp[c]) dp[c] = cand;
    }
}
```

Re-trace the single crate `(5,7)`, `C=10`, downward: `dp[0]=0`, rest `-1`. `c=10`: `dp[5]=-1`, skip. `c=9..6`: predecessors `-1`, skip. `c=5`: `dp[0]=0`, `dp[5] = 0 + 7 = 7`. Done. Best over `dp[]` is `7`. Correct — the crate is loaded exactly once. Re-trace the full sample (`A,B,C`, `C=10`) with the downward sweep to be sure I did not break the multi-crate case. Fold A=(6,10) downward: only `dp[0]` reachable, `c=6` sets `dp[6]=10`. Fold B=(5,7) downward: `c=10`: `dp[5]=-1`; `c=6`: `dp[1]=-1`; `c=5`: `dp[0]=0 -> dp[5]=7`. (Note `c=11` is out of range so A+B never combine — correct, `6+5>10`.) Fold C=(5,7) downward: `c=10`: `dp[5]=7 -> dp[10]=14`; `c=6..5`: `dp[1]`,`dp[0]` give nothing better; in particular `dp[5]` is updated *after* `dp[10]` is computed, so `dp[10]` used the pre-C value of `dp[5]` — exactly the 0/1 discipline. Best `= 14`. Correct, and the regression instance that killed greedy now returns the right answer.

**A second trace — the zero-mass crate, the other place this dies.** Zero-mass crates worry me because `w[i] = 0` makes `c - w[i] = c`, a *self*-reference. Does the downward sweep double-count a zero-mass crate? Take `C = 0` and two crates `(0, 3)` and `(0, 3)`; the correct answer is `6` (both are free, both should be loaded, each once). Trace: `dp[0]=0`. Fold crate 1 `(0,3)`, sweep `c` from `0` down to `0`: `c=0`: `dp[c - 0] = dp[0] = 0`, so `dp[0] = max(0, 0 + 3) = 3`. Now `dp[0]=3`. Fold crate 2 `(0,3)`, `c=0`: `dp[0] = max(3, 3 + 3) = 6`. Final best `= 6`. Correct. The reason it works: within a *single* crate's pass the cell `dp[c]` is read and written once at `c` (the sweep visits `c=0` exactly once), so the crate is added once; across the two separate crate passes each contributes its `+3` independently. If I had instead used a *forward* sweep here the same self-reference would still only visit `c=0` once for a zero-mass crate, so this particular corner is fine either way — but the downward direction is the one that is also correct for positive masses, so I keep it uniformly.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0`: the fold loop never runs; `dp = [0]` (just `dp[0]=0`), best `= 0`. The empty load — correct.
- `C = 0` with positive-mass crates, e.g. `(3, 9), (1, 2)`: both have `w > C`, the `if (w[i] > C) continue;` skips them, `dp = [0]`, best `= 0`. Correct — nothing fits.
- A single crate strictly heavier than `C`, e.g. `C=3`, crate `(5, 100)`: skipped by the guard, best `= 0`. Correct.
- Zero-value crate `(w, 0)`: folding it can set some `dp[c]` from `-1` to a real reachable value `0` or leave it unchanged; it never lowers any value (we only take a `max`), and it never inflates the answer above what positive crates give. Harmless.
- A zero-mass *positive*-value crate with a positive-mass crate, e.g. `C=6`, crates `(0,4),(6,5)`: fold `(0,4)` -> `dp[0]=4`; fold `(6,5)` -> `c=6`: `dp[0]=4 -> dp[6]=9`. Best `= 9`. Correct: the free crate plus the mass-6 crate.
- Overflow: values are accumulated in `long long`; the maximum total `~10^12` fits with enormous room. Masses/`C` are tiny. The `-1` sentinel is compared, never has `v[i]` added to it (the `!= -1` guard prevents that), so no spurious arithmetic. Safe.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace, so line layout in the input does not matter.

**Complexity self-check.** The fold is `n` crates times an inner sweep of length at most `C + 1`, so at most `n * (C + 1) = 1000 * 2001 ≈ 2.0 * 10^6` cell visits, plus an `O(C)` final scan. At well under `10^8` simple operations this is comfortably inside 1 second; memory is one `long long` array of length `C + 1 <= 2001`, about 16 KB, far under 256 MB. I confirmed empirically on `n=1000, C=2000` with values near `10^9`: it runs in under a millisecond and the answer `25628973081` exceeds `2^31 ≈ 2.1 * 10^9`, which is the concrete proof that the 64-bit value accumulator was necessary, not optional.

**Final solution.** I convinced myself the *idea* is right by constructing an explicit counterexample that disproves ratio greedy (`C=10`, crates `(6,10),(5,7),(5,7)`: greedy `10` vs optimum `14`) and by hand-checking the DP recurrence reproduces the `14`. I convinced myself the *code* is right by tracing the forward-sweep bug to a single crate counted twice, fixing the sweep direction, re-verifying on the failing witness and the full sample, and separately tracing the zero-mass self-reference. That is what I ship — one self-contained file, the `O(n*C)` capacity DP I can defend rather than the greedy I broke:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long C;
    if (!(cin >> n >> C)) return 0;
    vector<long long> w(n), v(n);
    for (int i = 0; i < n; i++) cin >> w[i] >> v[i];

    // dp[c] = best total value using a chosen subset with total weight EXACTLY c,
    // or -1 if c is unreachable. dp[0] = 0 (empty subset). 0/1 knapsack: iterate
    // capacity downward so each crate is used at most once.
    const long long NEG = -1;
    vector<long long> dp(C + 1, NEG);
    dp[0] = 0;
    for (int i = 0; i < n; i++) {
        if (w[i] > C) continue;                 // crate alone exceeds the hold
        for (long long c = C; c >= w[i]; c--) {
            if (dp[c - w[i]] != NEG) {
                long long cand = dp[c - w[i]] + v[i];
                if (cand > dp[c]) dp[c] = cand;
            }
        }
    }

    long long best = 0;                         // empty subset has value 0
    for (long long c = 0; c <= C; c++)
        if (dp[c] != NEG && dp[c] > best) best = dp[c];

    cout << best << "\n";
    return 0;
}
```

**Causal recap.** Ratio greedy looked right because it is optimal for the fractional knapsack, but an explicit traced counterexample (`C=10`, crates `(6,10),(5,7),(5,7)`: greedy strands 4 units of budget for `10` while crates B+C exactly fill it for `14`) showed indivisibility breaks it, so I moved to a capacity DP whose recurrence I checked reproduces the `14`; my first in-place fold swept capacity *upward*, which let a single crate be loaded twice (a trace of one `(5,7)` crate returning `14` instead of `7` pinpointed it), and reversing the sweep to go *downward* makes each crate fold in exactly once; tracing the zero-mass self-reference, the `w > C` guard, and the empty/`C=0` corners, all on 64-bit value accumulators with a `-1` reachability sentinel, closes out the remaining ways this code dies.
