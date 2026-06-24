**Reading the problem and pinning the contract.** I am loading one van. There are `n` parcels; parcel `i` takes `w[i]` units of volume and pays `v[i]` cents, and the van holds total volume `W`. I pick a subset of parcels with total volume at most `W` and maximize total payout, with each parcel usable at most once. Input on stdin is `n` and `W` on the first line, then `n` lines of `w[i] v[i]`; I print one integer. Before I pick any algorithm I fix the scale, because here the scale *is* the problem. `n <= 1000`, `0 <= v[i] <= 10^9`. If every parcel fits, the total payout is up to `1000 * 10^9 = 10^{12}`. A signed 32-bit integer tops out near `2.147 * 10^9`, so the answer overflows `int` by roughly three orders of magnitude. Every value, every accumulator, and the dp table itself must be 64-bit. I write that down now as a non-negotiable invariant: an `int` anywhere in the value arithmetic is a silent wrong answer on the big tests, not a crash. Volumes are at most `10^5` and `W` at most `10^5`, so volumes are safe in 32-bit, but I will still read everything into `long long` to avoid mixing widths by accident.

**Laying out the candidate approaches.** Two routes are on the table, and I want the one I can prove rather than the one that is shortest to type.

- *Greedy by value density.* Sort parcels by `v[i] / w[i]` descending and load the densest that still fits. `O(n log n)`, a handful of lines. This is exactly optimal for the *fractional* knapsack, where I can load a fraction of a parcel to fill the van to the brim. But here parcels are whole; I cannot take 0.6 of a parcel. I suspect greedy breaks, but suspicion is not proof, so I will try to break it with a concrete instance before I discard it.
- *Capacity dynamic programming.* For each volume budget `c` from `0` to `W`, hold the best payout achievable using volume at most `c`, and fold parcels in one at a time. `O(n*W)` time, `O(W)` memory. With `n*W = 1000 * 10^5 = 10^8` that is a hundred million simple operations — comfortably under a second in C++ if the inner loop is tight. The two things I must get exactly right are the *update direction* (which is what enforces "each parcel at most once") and, as flagged above, the *number width*.

**Stress-testing greedy before committing.** Let me actually attack greedy. Take `W = 4` and two parcels: parcel A with `(w, v) = (1, 2)` and parcel B with `(w, v) = (4, 7)`. Densities are `A = 2.0` and `B = 1.75`, so greedy loads A first (volume 1, payout 2), then tries B, which needs volume 4 but only 3 remains, so B is left behind. Greedy total: `2`. But loading B alone gives `7`, which fits exactly and is far better. So greedy's density ordering grabbed the high-density crumb and locked itself out of the one big parcel that filled the van. Greedy is wrong for 0/1, and I now see the mechanism precisely: density is the right tie-breaker only when leftover space can always be used (fractional case); with indivisible parcels a single bad early commit wastes the slack. Greedy is out. I go to the DP.

**Deriving the DP and checking the recurrence on paper.** I want `dp[c]` = the maximum payout achievable using a subset of the parcels considered so far whose total volume is at most `c`. Process parcels one at a time. When I bring in a new parcel with volume `w` and value `v`, for a budget `c >= w` I have two choices: do not use this parcel (payout stays `dp[c]`), or use it (payout becomes `dp_old[c - w] + v`, where `dp_old` is the table *before* this parcel was folded in, because the parcel is used at most once). So the transition for the new parcel is

`dp_new[c] = max(dp_old[c], dp_old[c - w] + v)` for `c >= w`, and `dp_new[c] = dp_old[c]` for `c < w`.

The base case before any parcel is `dp[c] = 0` for all `c`: with no parcels the best payout at any budget is zero (the empty load). The answer is `dp[W]` after all parcels are folded in. This is monotone in `c` by construction — a larger budget can always do at least as well — so reading `dp[W]` gives the best over all volumes up to `W`, which is what "total volume does not exceed `W`" asks for.

The subtlety that makes or breaks 0/1 knapsack is the phrase "`dp_old[c - w]`": the right-hand side must read the table *as it was before this parcel*, not the table I am currently updating. If I update budgets in increasing `c` and the table is in place, then by the time I reach `c` the entry `dp[c - w]` has already been overwritten *for the current parcel*, so I would be allowed to add the same parcel again — that is the *unbounded* knapsack, not 0/1. The standard fix is to iterate `c` from `W` downward to `w`: then `dp[c - w]` is still the pre-parcel value when I read it, because smaller indices have not been touched yet this round. I commit to the downward sweep and will verify it on a trace that distinguishes 0/1 from unbounded.

Let me confirm the recurrence by hand on the documented sample: `W = 10`, parcels `(3, 1000000000)`, `(4, 1500000000)`, `(5, 1200000000)`, `(2, 800000000)`. I expect `3300000000` from loading parcels 1, 2, 4 (volumes `3 + 4 + 2 = 9`). I will not trace all 11 budget cells by hand for four parcels — that is a lot of arithmetic — but I can sanity-check the claimed optimum: is there any fitting subset beating `3.3 * 10^9`? The only way to beat it is to include parcel 3 (value `1.2 * 10^9`) in place of something, but parcel 3 has volume 5; pairing it with parcels 1 and 4 gives volume `3 + 5 + 2 = 10` and value `1.0 + 1.2 + 0.8 = 3.0 * 10^9 < 3.3 * 10^9`, and pairing it with parcel 2 (volume `4 + 5 = 9`) gives `1.5 + 1.2 = 2.7 * 10^9`. Nothing beats `3.3 * 10^9`, and `1 + 2 + 4` fits, so the optimum is `3300000000`. The recurrence's job is to find that, and `dp[10]` should land there.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut, written quickly:

```
vector<int> dp(W + 1, 0);              // (A)
for (int i = 0; i < n; i++) {
    int w = wt[i], v = val[i];         // (B)
    for (int c = w; c <= W; c++) {     // (C) ascending
        int cand = dp[c - w] + v;
        if (cand > dp[c]) dp[c] = cand;
    }
}
cout << dp[W] << "\n";
```

Two things make me uneasy: the `int` dp at line (A)/(B), which I already flagged as fatal for big values, and the ascending loop at (C), which I argued enforces unbounded reuse rather than 0/1. Let me trace the smallest input that exposes the *reuse* bug first, because it changes which subset is chosen, not just the width. Take `W = 6` and a single parcel `(w, v) = (2, 5)`. The correct 0/1 answer is `5`: I have one copy, it fits, load it once. Trace the ascending loop. Start `dp = [0,0,0,0,0,0,0]` (indices 0..6). Parcel `(2,5)`. `c = 2`: `cand = dp[0] + 5 = 5`, `dp[2] = 5`. `c = 3`: `cand = dp[1] + 5 = 5`, `dp[3] = 5`. `c = 4`: `cand = dp[2] + 5 = 5 + 5 = 10`, `dp[4] = 10`. `c = 5`: `cand = dp[3] + 5 = 10`, `dp[5] = 10`. `c = 6`: `cand = dp[4] + 5 = 10 + 5 = 15`, `dp[6] = 15`. Final `dp[6] = 15`.

**Diagnosing the first bug.** The code returns `15`, which is *three copies* of the single parcel `(2, 5)` — `15 = 3 * 5`, three parcels of volume 2 in a budget of 6. That is illegal: there is exactly one parcel. The mechanism is exactly the one I predicted: at `c = 4` I read `dp[2]`, which I had *already raised to 5 in this same parcel's pass* (at `c = 2`), so adding `v` again loaded the parcel a second time; at `c = 6` reading `dp[4] = 10` loaded it a third. The ascending sweep lets `dp[c - w]` see this-parcel updates, which is the unbounded knapsack. I need the downward sweep so `dp[c - w]` is still the pre-parcel value. Switch (C) to `for (int c = W; c >= w; c--)`.

Re-trace `W = 6`, parcel `(2, 5)`, descending. Start `dp = [0,0,0,0,0,0,0]`. `c = 6`: `dp[4] + 5 = 0 + 5 = 5`, `dp[6] = 5`. `c = 5`: `dp[3] + 5 = 5`, `dp[5] = 5`. `c = 4`: `dp[2] + 5 = 5`, `dp[4] = 5`. `c = 3`: `dp[1] + 5 = 5`, `dp[3] = 5`. `c = 2`: `dp[0] + 5 = 5`, `dp[2] = 5`. Final `dp[6] = 5`. Correct — the parcel is loaded exactly once at every budget `>= 2`. The downward sweep is the fix, and it broke for the reason I named, which is the evidence I trust.

**Second trace — the overflow, on a concrete big input.** The reuse bug is fixed, but `dp` is still `int`. Now I deliberately exercise the value width, because it will *not* show up on the small ascending/descending trace above — those values are tiny. Take `W = 7` and two parcels `(3, 2000000000)` and `(4, 2000000000)`, both worth two billion cents. The correct 0/1 answer: both fit (`3 + 4 = 7 <= 7`), total `4000000000`. Trace the descending loop with an `int` dp. Folding parcel `(3, 2e9)`: `dp[3..7]` become `2000000000`. Folding parcel `(4, 2e9)`: at `c = 7`, `cand = dp[7 - 4] + 2000000000 = dp[3] + 2000000000 = 2000000000 + 2000000000`. Mathematically that is `4000000000`. But `cand` is declared `int`, and `4000000000 > 2^31 - 1 = 2147483647`. The addition of two `int`s overflows: `4000000000 - 2^32 = 4000000000 - 4294967296 = -294967296`. So `cand = -294967296`, which is *less* than `dp[7] = 2000000000`, so the `if (cand > dp[c])` never fires and `dp[7]` stays `2000000000`.

**Diagnosing the second bug.** The code returns `2000000000`, but the true answer is `4000000000`. The defect is precisely an `int` overflow in the value arithmetic: `dp[c - w] + v` exceeds `2^31 - 1`, wraps to a large negative number, and the `max` silently discards the correct larger payout. There is no crash, no warning — just a wrong, *too-small* answer, which is the worst kind because it looks plausible. This is exactly the trap the constraints were built around: `v[i]` up to `10^9` and up to `1000` parcels means real totals near `10^{12}`, and any 32-bit cell in the value path corrupts them. The cure is uniform: `dp` is `vector<long long>`, `v` and `w` are `long long`, and `cand` is `long long`, so the sum `dp[c - w] + v` is computed in 64-bit (max around `10^{12}`, which fits in `long long`'s `~9.2 * 10^{18}` with enormous headroom). I had already promised this in the contract step; the trace is what makes the promise concrete and proves the failure mode is silent.

Re-trace `W = 7`, parcels `(3, 2e9)` and `(4, 2e9)`, with `long long` everywhere, descending. Fold `(3, 2e9)`: `dp[3..7] = 2000000000`. Fold `(4, 2e9)`: `c = 7`: `dp[3] + 2e9 = 4000000000 > dp[7] = 2e9`, so `dp[7] = 4000000000`. `c = 6`: `dp[2] + 2e9 = 0 + 2e9 = 2e9`, not greater than `dp[6] = 2e9`, unchanged. `c = 5`: `dp[1] + 2e9 = 2e9`, unchanged. `c = 4`: `dp[0] + 2e9 = 2e9`, `dp[4]` was `2e9`, unchanged. Final `dp[7] = 4000000000`. Correct. The overflow is gone because the arithmetic is 64-bit; the descending sweep still keeps it 0/1.

**Adding the "item never fits" guard, and checking it does not break anything.** If `w[i] > W`, the inner loop `for (c = W; c >= w; c--)` never executes (its start `W` is already below `w`), so the parcel is correctly ignored even without a guard. But I add an explicit `if (w > W) continue;` for clarity and to avoid pathological loop setup. I must check this guard does not skip a *useful* parcel: a parcel with `w[i] > W` cannot be in any feasible subset (it alone exceeds the van), so skipping it is exactly correct. Zero-volume parcels (`w = 0`) are not skipped — the loop runs `c` from `W` down to `0`, and `dp[c - 0] + v = dp[c] + v` adds `v` whenever `v > 0`, loading the free parcel at every budget. That is correct: a volume-0 positive-value parcel should always be loaded. (The descending order still applies it once: reading `dp[c]` before this parcel touched it — wait, at `w = 0` the read index equals the write index, so I should check this does not double-count. Tracing `W = 1`, parcel `(0, 5)`, descending: `c = 1`: `dp[1] + 5 = 0 + 5 = 5`, set `dp[1] = 5`. `c = 0`: `dp[0] + 5 = 5`, set `dp[0] = 5`. Each cell is written once from its own pre-parcel value, so the free parcel is loaded exactly once per budget, payout `5`, not `10`. Correct.)

**Edge cases, deliberately, because this is where knapsack code dies.**
- `n = 0`: the parcel loop never runs, `dp` stays all zeros, output `dp[W] = 0`. The empty load — correct. (`cin >> n >> W` still reads `W`; even if input ends, the `if (!(cin >> n >> W)) return 0;` guard prints nothing, but the contract guarantees `n` and `W` are present, and `n = 0` with a `W` value gives `0`.)
- `W = 0`: `dp` has a single cell `dp[0] = 0`. Any parcel with `w > 0` is skipped by the guard. A parcel with `w = 0` and `v > 0` runs the loop at `c = 0` only and sets `dp[0] = v` (max over all such). Correct: with zero volume budget only free parcels can be loaded.
- All parcels too big (`w[i] > W` for all `i`): every parcel is skipped, output `0`. Correct — nothing fits.
- Ties in value or density: the DP compares payouts numerically and keeps the larger; ties resolve to the same payout, so the chosen subset may differ but the reported maximum is identical. Correct.
- Overflow: the value path is uniformly `long long`; the max total `~10^{12}` fits with a factor of `~10^6` of headroom. There are no negative sentinels to underflow (all payouts are `>= 0`, base case `0`). Volumes peak at `10^5`, safe in `long long`. Safe.
- Output: exactly one integer and a newline; `cin >>` skips arbitrary whitespace, so line/space layout of the input does not matter.

**Sanity-checking the derivation against the sample once more.** The contract sample (`W = 10`, four parcels) must yield `3300000000`. My DP folds all four parcels with the downward sweep and 64-bit cells; I argued above no fitting subset beats `1 + 2 + 4 = 3.3 * 10^9` and that triple fits in volume 9, so `dp[10]` reaches `3300000000`. The number itself exceeds `2^31 - 1`, so the sample doubles as a built-in overflow detector: any `int` solution prints something other than `3300000000` here. That is reassuring — the very first example would catch the most common mistake.

**Final solution.** I disproved greedy with a one-parcel-fills-the-van counterexample, derived the capacity DP, and then earned the two corrections by tracing: the ascending sweep loaded one parcel three times (fixed by sweeping `c` downward so `dp[c - w]` is the pre-parcel value), and the `int` cells silently wrapped a `4 * 10^9` total to a negative number that the `max` threw away (fixed by making the entire value path `long long`). The corners — `n = 0`, `W = 0`, oversized parcels, free parcels, ties, overflow — all check out. This is what I ship: one self-contained file, the `O(n*W)` 0/1 knapsack I can defend.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long W;
    if (!(cin >> n >> W)) return 0;

    vector<long long> wt(n), val(n);
    for (int i = 0; i < n; i++) cin >> wt[i] >> val[i];

    // dp[c] = maximum total value achievable using a capacity budget of exactly-at-most c.
    // 0/1 knapsack: each item used at most once -> iterate capacity downward.
    vector<long long> dp(W + 1, 0);
    for (int i = 0; i < n; i++) {
        long long w = wt[i], v = val[i];
        if (w > W) continue;                       // item never fits
        for (long long c = W; c >= w; c--) {
            long long cand = dp[c - w] + v;        // both operands long long -> no overflow
            if (cand > dp[c]) dp[c] = cand;
        }
    }

    cout << dp[W] << "\n";
    return 0;
}
```

**Causal recap.** Greedy by density looked right but a single counterexample (`W = 4`, parcels `(1, 2)` and `(4, 7)`: greedy loads the dense crumb for `2` and locks out the `7` parcel that fills the van) showed indivisible parcels make density ordering non-optimal, so I moved to the capacity DP `dp[c] = max(dp[c], dp[c - w] + v)`; the at-most-once rule requires `dp[c - w]` to be the pre-parcel value, which an ascending sweep violates — a trace of one parcel `(2, 5)` in budget `6` returning `15` (three copies) pinpointed it, and sweeping `c` from `W` downward fixes it; and because payouts reach `~10^{12}` while a 32-bit cell wraps at `2.1 * 10^9`, a second trace of two `2 * 10^9` parcels returning `2 * 10^9` instead of `4 * 10^9` exposed a silent `int` overflow in `dp[c - w] + v`, fixed by making the whole value path `long long`; the empty, zero-budget, oversized-parcel, free-parcel, and overflow corners then all resolve correctly, and the sample's `3300000000` (itself above `2^31 - 1`) doubles as an overflow tripwire.
