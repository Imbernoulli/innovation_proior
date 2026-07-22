I have `n` experiments, each with an energy cost `e[i]` and a science value `v[i]`, and one battery of `W` energy units. I pick a subset whose energy sums to at most `W`, maximizing total value, and crucially **each experiment can be run at most once** — there is a single copy of each instrument on board. Running nothing is allowed, so the answer is never below `0`. Input is `n W` then `n` pairs; I print one integer. Before any algorithm I fix the scale, because it dictates types: `n <= 2000`, `W <= 2*10^5`, `e[i] <= 10^5`, `v[i] <= 10^9`. A feasible selection can include up to `n = 2000` experiments, so the total value can reach `2000 * 10^9 = 2*10^12`. That is far past the 32-bit ceiling of `~2.1*10^9`, so the DP table and every accumulator must be 64-bit. I will use `long long` for value-carrying state. An `int` here is a silent wrong-answer on the large tests, so this is decided first and non-negotiable.

**Recognizing the shape and the candidate approaches.** This is 0/1 knapsack: "at most once" is the 0/1 rule, energy is weight, value is profit, the battery is capacity. The time budget is generous — `n*W = 2000 * 2*10^5 = 4*10^8` is fine for 1 second with a tight inner loop — so I do not need anything clever; I need the *right* recurrence. Two implementations are on the table:

- *2D table.* `dp[i][c]` = best value using the first `i` items within energy `c`. Transition `dp[i][c] = max(dp[i-1][c], dp[i-1][c-e[i]] + v[i])`. Memory `O(n*W) = 4*10^8` longs — way over 256 MB. Unusable as written.
- *1D rolling array.* The well-known space optimization: keep a single `dp[c]` and update it item by item in place. This is "the standard knapsack DP" everyone reaches for, `O(n*W)` time, `O(W)` memory. The catch — and the whole point of this problem — is the **loop direction**. There are two versions floating around, one for unbounded supply and one for at-most-once, differing only in whether the capacity loop runs ascending or descending. I have to get that exactly right, and I do not trust my memory of which is which; I will derive it and then trace it.

**Deriving the 1D recurrence from the 2D one.** Start from the correct 2D transition `dp[i][c] = max(dp[i-1][c], dp[i-1][c-e[i]] + v[i])`. I want to overwrite a single array `dp[c]` so that *before* I process item `i` it holds the row `i-1`, and *after* it holds row `i`. The danger is the term `dp[i-1][c-e[i]]`: it must read the value from the **previous** row (item `i` not yet used), i.e. the *old* `dp[c-e[i]]`. If, while updating `dp` in place, I have already overwritten `dp[c-e[i]]` with a row-`i` value, then `dp[c-e[i]]` may already contain item `i`, and adding `v[i]` again uses item `i` twice. So the order in which I touch capacities decides whether `dp[c-e[i]]` is still "old". If I sweep `c` from small to large, then `c-e[i] < c` is touched *before* `c`, so `dp[c-e[i]]` is already the new row — reuse is allowed (that is unbounded knapsack). If I sweep `c` from `W` down to `e[i]`, then `c-e[i] < c` is touched *after* `c`, so when I read `dp[c-e[i]]` it is still the old row — each item used at most once (that is 0/1 knapsack, what I want). So: **descending** capacity. I will verify this claim numerically rather than trust the derivation.

**Numeric self-check of the direction claim on a singleton.** Take a single experiment `(e, v) = (2, 5)` and `W = 4`, where the right answer is obviously `5` (one copy, energy `2`). Run the *ascending* update by hand on `dp = [0,0,0,0,0]`: `c=2`: `dp[2] = max(0, dp[0]+5) = 5`. `c=3`: `dp[3] = max(0, dp[1]+5) = 5`. `c=4`: `dp[4] = max(0, dp[2]+5)` — but `dp[2]` was just set to `5` this same sweep, so `dp[4] = 0 + ... = 5 + 5 = 10`. Ascending gives `dp[4] = 10`, i.e. the experiment run twice — wrong for at-most-once. Now the *descending* update on a fresh `dp`: `c=4`: `dp[4] = max(0, dp[2]+5) = 0+5 = 5` (`dp[2]` still old `0`). `c=3`: `dp[3] = max(0, dp[1]+5) = 5`. `c=2`: `dp[2] = max(0, dp[0]+5) = 5`. Descending gives `dp[4] = 5` — correct. The derivation holds: descending for 0/1. This single check already shows the "standard" ascending loop is the wrong tool here.

**Why the constraints make the trap bite.** The reason this is a real hazard and not a triviality: `e[i]` can be as small as `1` while `W` is up to `2*10^5`, so a single cheap high-value experiment, if it were allowed to repeat, could be run tens of thousands of times and the ascending answer would balloon. The wrong variant does not merely err in a corner — it diverges hugely. So I cannot lean on "it usually agrees"; I must use the at-most-once update.

**First implementation — write it, then trace, because I half-remember the direction.** Despite the derivation, muscle memory wants to type the ascending loop (it is the first knapsack snippet most people memorize). Suppose my first cut is:

```
vector<long long> dp(W + 1, 0);
for (int i = 0; i < n; i++) {
    if (e[i] > W) continue;
    for (long long c = e[i]; c <= W; c++)              // ASCENDING (suspect)
        dp[c] = max(dp[c], dp[c - e[i]] + v[i]);
}
```

I will trace it on the documented sample: `n = 4`, `W = 10`, experiments `(3,8), (3,8), (4,9), (2,5)`; the intended answer is `25` (run the first three: energy `3+3+4 = 10`, value `8+8+9 = 25`). I expand the `dp` array (indices `0..10`) after each item under the ascending loop.

- After item `(3,8)`: sweeping `c = 3..10` upward, `c=3 -> 8`, `c=6 -> dp[3]+8 = 16`, `c=9 -> dp[6]+8 = 24`. So `dp = [0,0,0,8,8,8,16,16,16,24,24]`. Already `dp[9] = 24` — that is experiment 0 run **three** times for value `24`. The bug is visible after the very first item.
- After item `(3,8)`: identical numbers (the second `(3,8)` cannot improve states the first already inflated), `dp = [0,0,0,8,8,8,16,16,16,24,24]`.
- After item `(4,9)`: `dp = [0,0,0,8,9,9,16,17,18,24,25]`.
- After item `(2,5)`: `dp = [0,0,5,8,10,13,16,18,21,24,26]`. Final `dp[10] = 26`.

**The bug, named precisely.** The ascending loop printed `26`, but the true optimum is `25`. The defect is exactly the reuse the derivation warned about: within one item's sweep, `dp[c - e[i]]` was already updated *this same pass*, so adding `v[i]` again stacked another copy of the same experiment. The clearest fingerprint is `dp[9] = 24 = 3 * 8` after a single `(3,8)` item — three runs of one instrument, which the hardware forbids. My ascending code is solving *unbounded* knapsack; it is the wrong standard algorithm for this exact variant. I will not "appeal to the standard solution"; the trace settled it.

**Fix and re-verification.** Flip the inner loop to descending so `dp[c - e[i]]` is read before it is overwritten this pass:

```
vector<long long> dp(W + 1, 0);
for (int i = 0; i < n; i++) {
    if (e[i] > W) continue;
    for (long long c = W; c >= e[i]; c--)              // DESCENDING (0/1)
        dp[c] = max(dp[c], dp[c - e[i]] + v[i]);
}
```

Re-trace the sample with the descending update, expanding `dp[0..10]` after each item:

- After `(3,8)`: only `dp[c] = dp[c-3]+8` for `c = 10..3` with old `dp` all zero, so just `dp[3..10] = 8`: `dp = [0,0,0,8,8,8,8,8,8,8,8]`. No reuse — `dp[9]` is `8`, not `24`. Good.
- After the second `(3,8)`: now `dp[6] = dp[3]+8 = 16`, and `dp[c] = 16` for `c >= 6`; `dp[3..5] = 8`. `dp = [0,0,0,8,8,8,16,16,16,16,16]`. Two distinct `(3,8)` experiments summed to `16` — legal, they are separate instruments.
- After `(4,9)`: `dp = [0,0,0,8,9,9,16,17,17,17,25]`. `dp[10] = dp[6]+9 = 16+9 = 25` — experiments 0,1,2.
- After `(2,5)`: `dp = [0,0,5,8,9,13,16,17,21,22,25]`. Final `dp[10] = 25`.

Descending returns `25`, matching the intended optimum, and no instrument is ever counted twice (the telltale `24` after one item is gone). The fix addresses the exact cause the trace exposed.

**Cross-checking the fix against a brute force where the two variants diverge hardest.** One sample is not enough; I want a case engineered so reuse would help a lot, then compare the descending DP, the ascending DP, and an exhaustive subset enumeration. Take three cheap, high-value experiments `(2,18), (2,8), (2,15)` with `W = 6`. The exhaustive answer: every item costs `2`, three of them total energy `6 == W`, so the only feasible "all three" selection has value `18 + 8 + 15 = 41`, and no two-item subset beats it, so the optimum is `41`. Now the **descending** DP, `dp[0..6]`: after `(2,18)` -> `dp = [0,0,18,18,18,18,18]`; after `(2,8)` -> `dp[c] = max(dp[c], dp[c-2]+8)` swept high-to-low gives `dp = [0,0,18,18,26,26,26]`; after `(2,15)` -> `dp = [0,0,18,18,33,33,41]`, so `dp[6] = 41`. It matches the brute force exactly — each instrument used once. The **ascending** DP on the same input climbs to `dp = [0,0,18,18,36,36,54]`, answer `54`: at `c=4` it does `dp[2]+18 = 18+18 = 36` (experiment 0 *twice*) and at `c=6` `dp[4]+18 = 36+18 = 54` (experiment 0 *three* times). So on a case where the variants differ by `54 - 41 = 13`, the descending DP agrees with exhaustive enumeration and the ascending one does not. This is the decisive cross-check: I am not trusting the loop direction on faith, I am matching an independent oracle.

**A second self-verify episode: the off-by-one `e[i] > W` guard.** A subtle second bug lurks in the guard and bounds. My first instinct for the guard was `if (e[i] >= W) continue;` (skip items "as heavy as the battery"). Trace that on `n = 1`, `W = 4`, experiment `(4, 9)`: the experiment fits exactly (energy `4 == W`), so the answer must be `9`. But `e[0] = 4 >= W = 4` would `continue` and skip it, printing `0` — wrong. The correct guard is `e[i] > W` (skip only items strictly heavier than the battery). With `e[i] > W` false here, the loop runs `c = 4..4`, `dp[4] = max(0, dp[0]+9) = 9`. Correct. Equally, the inner loop bound `c >= e[i]` (not `c >= e[i]+1` or `c > e[i]`) is what lets an item that exactly fills remaining capacity be taken: at `c = e[i]`, `dp[c-e[i]] = dp[0]` is read, which is valid. I confirm there is no out-of-range access: `c` ranges in `[e[i], W]` and `c - e[i]` ranges in `[0, W - e[i]]`, both within `[0, W]`. The `e[i] > W` skip also guarantees `e[i] <= W` whenever the loop body runs, so the loop is non-empty and well-formed.

**Edge cases, deliberately.**
- `W = 0`: the battery is empty. Every `e[i] >= 1 > 0 = W`, so every item is skipped by the guard; `dp` stays all zero; `dp[0] = 0`. Correct — nothing can run.
- An item with `e[i] > W` among others: it is skipped and never pollutes `dp`; the rest are scheduled normally. Correct.
- Value `0` experiments (`v[i] = 0`): they can be "taken" but add nothing, and `max` never prefers them spuriously; the answer is driven by positive-value items. Correct.
- All items too heavy: `dp[W]` stays `0`. Correct — run nothing.
- `n = 1` with a fitting item: handled by the guard trace above (`9`).
- Overflow: `dp` is `vector<long long>`; the worst total `2000 * 10^9 = 2*10^12` fits in `long long` (`~9.2*10^18` ceiling) with enormous headroom. No sentinel arithmetic is involved — `dp` starts at `0` and only ever increases by non-negative `v[i]`, so no underflow either.
- Memory: one `long long` array of size `W+1 <= 2*10^5 + 1`, about `1.6` MB — well under `256` MB. (The 2D table I rejected would have been `4*10^8` longs.)
- Output: exactly one integer and a newline; `cin >>` skips arbitrary whitespace, so parsing is format-agnostic.

**Final solution.** I established the idea is 0/1 knapsack, derived that the 1D rolling array must sweep capacity **descending**, and — rather than trust "the standard knapsack snippet" — traced the ascending version on the sample, watched it count one experiment three times (`dp[9] = 24` after a single `(3,8)`) and return `26` instead of `25`, then flipped the loop and re-traced to confirm `25` with no reuse. The guard `e[i] > W` (strict) and the bound `c >= e[i]` were each pinned by their own trace. This is what I ship — one self-contained file, the descending 1D 0/1 knapsack:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long W;
    if (!(cin >> n >> W)) return 0;          // no data -> nothing scheduled
    vector<long long> e(n), v(n);            // e[i] = energy cost, v[i] = science value
    for (int i = 0; i < n; i++) cin >> e[i] >> v[i];

    // 0/1 knapsack: each experiment scheduled at most once, total energy <= W.
    // dp[c] = best total value using energy budget exactly-bounded-by c.
    vector<long long> dp(W + 1, 0);
    for (int i = 0; i < n; i++) {
        if (e[i] > W) continue;              // never fits, skip to avoid touching dp out of range
        // DESCENDING capacity: each item contributes to a strictly smaller earlier state,
        // so within this i the item is used at most once.
        for (long long c = W; c >= e[i]; c--) {
            long long cand = dp[c - e[i]] + v[i];
            if (cand > dp[c]) dp[c] = cand;
        }
    }

    cout << dp[W] << "\n";
    return 0;
}
```

**Causal recap.** The problem is 0/1 knapsack (each experiment once), so the memory-saving 1D rolling array — the algorithm everyone reaches for — is only correct in its *descending* form; the seductive ascending form is the *unbounded* knapsack and silently reuses items. I derived the direction from the 2D transition (reading `dp[c-e[i]]` must hit the previous row), confirmed it numerically on a singleton (`(2,5)`, `W=4`: ascending `10` vs correct `5`), then traced the ascending loop on the documented sample where it returned `26` and exposed its fingerprint `dp[9] = 24 = 3*8` after one `(3,8)` item; flipping to descending re-traced to the correct `25` with no item counted twice. The strict guard `e[i] > W` (caught by a `W`-exact item printing `0`), the bound `c >= e[i]`, and a 64-bit `dp` (totals up to `2*10^12`) close out the empty-budget, too-heavy, zero-value, single-item, and overflow corners.
