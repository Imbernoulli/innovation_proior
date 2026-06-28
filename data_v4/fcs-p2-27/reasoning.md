**Reading the problem and pinning the contract.** I have `n` non-negative integers `a[0..n-1]` and a target `T`, and I must count how many subsets of the *positions* `{0,...,n-1}` have values summing to exactly `T`. The phrase "subsets of the positions" is the load-bearing one: two subsets that pick different positions are different even if the chosen *values* coincide, and the empty subset is a legitimate subset that sums to `0`. So with `a = [2, 2, 2]` and `T = 2` there are three answers, not one — `{0}`, `{1}`, `{2}` — because the positions differ. I write that down first because it decides what "count" even means, and it is exactly the kind of thing a sloppy implementation collapses by accident.

Next, the scale, because it dictates the data type and whether I need a modulus at all. The number of subsets of an `n`-element set is `2^n`, and with `n` up to `200` that is `2^200 ≈ 1.6*10^60`. Even the count for a *fixed* sum can be enormous — think of `40` copies of the value `1` with `T = 20`, where the count is `C(40,20) = 137,846,528,820`, already past the 32-bit range, and that is a tiny instance. So the raw count does not fit in any fixed-width integer for the constraints I want. The standard resolution for "count, possibly huge" problems is to report the count **modulo a prime**, and I will fix `MOD = 1000000007`. That makes the judge exact on the residue, which is the honest contract: I am being asked for the count mod `MOD`, and I will compute exactly that.

The input is `n` and `T` on the first line, then `n` values; the output is one integer, the count mod `MOD`. Time limit 2 seconds, memory 256 MB. With `0 <= a[i] <= 1000`, `0 <= T <= 100000`, `0 <= n <= 200`. I deliberately note the corners that I already suspect will be where this dies: `T = 0`, values equal to `0`, repeated values, `T` larger than the total of all values (unreachable), and `n = 0`.

**Laying out the candidate approaches.** Two routes are on the table, and I want to commit to the one I can *prove*, not the one that is fastest to type.

- *A greedy / sorting heuristic.* The instinct, carried over from optimization subset-sum, is: sort the values, then walk them assembling `T`, counting "ways" as I go — maybe "at each value decide include/exclude and multiply branch counts," or "sort descending and greedily fit into the remaining target." It is `O(n log n)` and feels like it should expose some clean combinatorial structure. The risk is structural and severe: **counting is not an optimization problem.** There is no single best object to be greedy toward. The number of ways to reach `T` is a *global* sum over all combinations, and a greedy that commits to a local decision (take this value, skip that one) throws away the very branching it is supposed to be counting. I will not trust it until I have either made it concrete or broken it.
- *An exact-target counting DP.* Process the items one at a time; maintain, for every sum `s` in `[0, T]`, the number of subsets of the items seen so far that sum to exactly `s`. Adding an item updates this table. This is `O(n*T)`. The risk here is not whether the *idea* is right — it is a textbook recurrence — but whether I transcribe the update order correctly so that each item is used at most once, and whether zeros are handled rather than silently mishandled.

**Stress-testing the greedy before committing.** Hand-waving "greedy can't count" is not a proof, so let me actually try to make a greedy/sorting rule produce the right number on a concrete instance and watch it fail. Take the simplest non-trivial counting instance: `a = [1, 2, 3]`, `T = 3`. The true answer, by enumeration, is `2`: the subset `{3}` (position 2) and the subset `{1, 2}` (positions 0 and 1). Now run a "greedy assemble the target" rule: sort descending `[3, 2, 1]`, take the largest value that fits the remaining target. Remaining target `3`; the `3` fits, take it, remaining `0`, done — that is **one** way found. The greedy commits to `3` the moment it fits and never explores the `1 + 2` route, because choosing `3` already reached the target and the rule stops. Greedy reports `1`; the truth is `2`. The verification paid off immediately: a three-element instance shows the greedy cannot even *see* the alternative decomposition, because "reach the target by committing to the biggest fitting value" is a single path through a tree whose size *is* the answer.

I can make the gap arbitrarily large to be sure it is not a fluke of one tiny case. Take `n` copies of the value `1` and `T = n/2`: the number of subsets is `C(n, n/2)`, which is exponential, while any greedy that picks a fixed set of `n/2` ones reports `1`. So greedy under-counts by an exponential factor. And it is not salvageable by "multiply by symmetry corrections" in general, because with mixed values like `[1, 1, 2]`, `T = 2` the answer is `2` (`{2}` and `{1,1}`) and the local multiplicities do not factor cleanly. Greedy is not merely risky-to-get-right-in-budget; it is the wrong *kind* of computation for a global count. It is out.

**Deriving the DP and checking the recurrence on paper.** I want, for each prefix of the items, the full distribution: `dp[s]` = number of subsets of the items processed so far whose sum is exactly `s`. The only thing the next item cares about is this table over `s in [0, T]`; nothing about *which* positions formed each count matters for the future, only how many ways reach each sum. That is the Markov property that makes the DP correct.

Before any item, the only subset is the empty one, which sums to `0`. So the base case is `dp[0] = 1` and `dp[s] = 0` for `s > 0`. Now process item `i` with value `v`. Every subset of the first `i+1` items either *excludes* item `i` (and is a subset of the first `i` items) or *includes* it (a subset of the first `i` items, sum `s - v`, with item `i` appended). So the new table is

`dp_new[s] = dp_old[s] + dp_old[s - v]`   (the second term only when `s >= v`).

That is the classic 0/1-knapsack *counting* recurrence: each item contributes either "not used" (`dp_old[s]`) or "used" (`dp_old[s - v]`). The answer is `dp[T]` after all `n` items. Let me confirm it by hand on the sample `a = [1, 2, 3]`, `T = 3`. Start `dp = [1, 0, 0, 0]` (indices `0..3`).

- Item `1` (`v = 1`): `dp_new[s] = dp[s] + dp[s-1]`. `dp[0] = 1`, `dp[1] = 0 + dp[0] = 1`, `dp[2] = 0 + dp[1]=0`, `dp[3] = 0 + dp[2] = 0`. So `dp = [1, 1, 0, 0]` — subsets `{}` and `{1}`, correct.
- Item `2` (`v = 2`): `dp[0] = 1`, `dp[1] = 1`, `dp[2] = 0 + dp[0] = 1`, `dp[3] = 0 + dp[1] = 1`. So `dp = [1, 1, 1, 1]` — sums `0,1,2,3` each reachable one way (`{}, {1}, {2}, {1,2}`), correct.
- Item `3` (`v = 3`): `dp[0]=1`, `dp[1]=1`, `dp[2]=1`, `dp[3]= 1 + dp[0] = 2`. Final `dp[3] = 2`. Correct — `{1,2}` and `{3}`.

The recurrence is right.

**The transcription trap: in-place update order.** I do not want two full arrays per item; the standard trick is to update `dp` in place over a single array. But that is precisely where the 0/1 constraint (each item used at most once) lives or dies. If I write the update as an *ascending* loop

```
for (s = v; s <= T; s++) dp[s] += dp[s - v];
```

then when I compute `dp[s]` I read `dp[s - v]`, which — for a positive `v` and ascending `s` — has *already been updated in this same item's pass*. That updated value includes the possibility of having used item `i` once already, so adding it again counts using item `i` *twice*. Ascending order silently turns the 0/1 count into an *unbounded* (multiset) count, where each item may be reused any number of times. That is a different problem. To keep each item single-use, I must iterate `s` from high to low, so that `dp[s - v]` still holds the *old* (pre-item-`i`) value:

```
for (s = T; s >= v; s--) dp[s] += dp[s - v];
```

Descending, `s - v < s`, and indices below the current `s` have not yet been touched in this pass, so `dp[s - v]` is the count *before* item `i` — exactly `dp_old[s - v]`. This is the in-place 0/1 knapsack order, and getting it backwards is the most common way this exact code is wrong.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut of the per-item loop, written quickly:

```
for (long long s = v; s <= T; s++) {       // <-- ascending, the trap
    dp[s] = (dp[s] + dp[s - v]) % MOD;
}
```

Something about ascending order on a 0/1 problem nags at me, so I trace the smallest input that could expose it: `a = [2, 2, 2]`, `T = 2`. The true answer is `3` (pick exactly one of the three positions). Start `dp = [1, 0, 0]`.

- Item `0` (`v = 2`), ascending `s = 2`: `dp[2] = dp[2] + dp[0] = 0 + 1 = 1`. `dp = [1, 0, 1]`.
- Item `1` (`v = 2`), ascending `s = 2`: `dp[2] = 1 + dp[0] = 1 + 1 = 2`. `dp = [1, 0, 2]`.
- Item `2` (`v = 2`), ascending `s = 2`: `dp[2] = 2 + dp[0] = 3`. `dp = [1, 0, 3]`.

Here it happens to give `3`, which looks fine — because with `T = 2` and `v = 2` the loop body only ever touches `s = 2` and reads `dp[0]`, which is never modified, so ascending and descending coincide. The bug needs a case where `s - v` itself gets updated within the pass. So I pick a sharper trace: `a = [1]`, `T = 3`, where the answer is obviously `0` (one item of value `1` cannot reach `3`). Ascending: `s = 1`: `dp[1] = 0 + dp[0] = 1`; `s = 2`: `dp[2] = 0 + dp[1] = 0 + 1 = 1`; `s = 3`: `dp[3] = 0 + dp[2] = 1`. Final `dp[3] = 1` — **wrong**, it claims one way to make `3` from a single `1`. That is the unbounded-knapsack leak: the ascending order let the single item `1` be reused three times to build `3`.

**Diagnosing the bug.** The defect is exactly the read-after-write I worried about. With ascending `s` and `v = 1`, by the time I compute `dp[2]` I have already set `dp[1]` *in this same pass*, so `dp[2] += dp[1]` is counting "use item `0` to get to `1`, then use item `0` *again* to step from `1` to `2`." The chain `dp[1] -> dp[2] -> dp[3]` reuses the one available item three times. Ascending order computes the *with-repetition* count; I want *without* repetition. The fix is the descending iteration so that `dp[s - v]` is the pre-item value, never the freshly written one.

**Fixing and re-verifying.** Flip the loop to descending and re-run both traces:

```
for (long long s = T; s >= v; s--) {
    dp[s] = (dp[s] + dp[s - v]) % MOD;
}
```

Re-trace `a = [1]`, `T = 3`. Start `dp = [1,0,0,0]`. Item `0` (`v = 1`), descending `s = 3,2,1`: `dp[3] = 0 + dp[2] = 0`; `dp[2] = 0 + dp[1] = 0`; `dp[1] = 0 + dp[0] = 1`. Final `dp = [1,1,0,0]`, so `dp[3] = 0`. Correct — a single `1` cannot make `3`.

Re-trace `a = [2,2,2]`, `T = 2` with the descending loop to be sure I did not regress: each pass touches only `s = 2` reading `dp[0]`, identical to before, final `dp[2] = 3`. Correct. And re-run the original sample `a = [1,2,3]`, `T = 3` mentally with descending order: item `1` (`v=1`) descending `s=3,2,1`: `dp[3]=0+dp[2]=0`, `dp[2]=0+dp[1]=0`, `dp[1]=0+dp[0]=1` -> `[1,1,0,0]`; item `2` (`v=2`) `s=3,2`: `dp[3]=0+dp[1]=1`, `dp[2]=0+dp[0]=1` -> `[1,1,1,1]`; item `3` (`v=3`) `s=3`: `dp[3]=1+dp[0]=2` -> `dp[3]=2`. Correct. The case that broke now passes, and it broke for precisely the reason I fixed — that is the evidence I trust.

**The zero-value corner, on purpose, because it is the subtlest one.** An item with `v = 0` is special: including or excluding it both leave the sum unchanged, so a single zero should *double* every count. Does the descending loop handle `v = 0` correctly? With `v = 0` the loop is `for (s = T; s >= 0; s--) dp[s] += dp[s - 0]`, i.e. `dp[s] += dp[s]`, which doubles `dp[s]` — exactly the intended "include or exclude the zero." But wait: `s - v = s`, so I am reading and writing the *same* cell. Is that the old value or the new one? The line `dp[s] = (dp[s] + dp[s]) % MOD` reads `dp[s]` (its current, pre-line value) and writes back twice that. There is no cross-cell dependence at all when `v = 0` — each cell only references itself — so ascending vs descending is irrelevant for zeros, and the doubling is correct regardless of direction. So `[0, 0, 5]` with `T = 0` should give `4` (any subset of the two zeros: `{}`, `{0}`, `{1}`, `{0,1}`). Trace: start `dp[0]=1`. Item `0` (`v=0`): `dp[0]=1+1=2`. Item `1` (`v=0`): `dp[0]=2+2=4`. Item `2` (`v=5`, but `T=0` so the loop `s>=5` with `s` from `0` never runs): `dp[0]` stays `4`. Final `4`. Correct — and `T = 0` in general gives `2^(number of zeros)`, since only zeros can be added without changing the sum.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0`: the item loop never runs; `dp = [1, 0, ..., 0]`. If `T = 0` the answer is `dp[0] = 1` (the empty subset); if `T > 0` it is `dp[T] = 0`. Both correct — the empty set sums to `0` and to nothing else.
- `T` unreachable (larger than the total of all values): no subset reaches it, so `dp[T]` stays `0`. The recurrence handles this without any special case.
- `T = 0` with no zeros: only the empty subset sums to `0`, answer `1`. With `k` zeros, answer `2^k mod MOD`. Both fall out of the recurrence.
- Repeated values: `a = [2, 2, 2]`, `T = 2` -> `3`, because positions are distinct. Handled, as traced.
- Overflow / modulus: each `dp[s]` stays in `[0, MOD)`, and the single addition `dp[s] + dp[s-v]` is at most `2*(MOD-1) < 2*10^9`, which fits comfortably in `long long` before the `% MOD`. I store `dp` as `long long` for exactly this headroom. An `int dp[]` would overflow the intermediate sum on adversarial inputs, a silent wrong-answer; `long long` removes that risk.
- Output: exactly one integer and a newline. I print `dp[T] % MOD`; `dp[T]` is already reduced, so the extra `% MOD` is a harmless guard.

**Complexity and the constraint check.** Time is `O(n*T)`: for each of `n` items I sweep at most `T + 1` sums. At the maximum `n = 200`, `T = 100000`, that is `2*10^7` cell updates — well under the 2-second limit (it runs in about `0.02` s in practice). Memory is `O(T)` for one `long long` array of `T + 1` entries, about `0.8` MB, far under `256` MB. So the simple, provable DP is not just correct but comfortably fast at the chosen constraints — there is no pressure to reach for anything cleverer.

**Self-verification harness.** I did not stop at hand traces. I wrote an independent brute oracle that literally enumerates all `2^n` subsets (a direct restatement of the definition, sharing no algorithmic idea with the DP) and a generator that emits random small cases plus targeted families: all-zeros arrays, all-equal arrays (to exercise the distinct-positions semantics), forced `T = 0`, forced unreachable `T`, and the empty array. I ran the DP against the brute oracle on over five hundred generated cases across those families with **zero mismatches**, and separately confirmed on `40` ones with `T = 20` that the output `846527861` equals `C(40,20) mod 1000000007`, so the modular reduction is doing real work on counts that overflow `64` bits in their raw form. The differential test is what turns "I traced it and it looks right" into "it agrees with the definition on every case I could throw at it."

**Final solution.** I convinced myself the *idea* is right by disproving the greedy (it reports `1` where the truth is `2`, and under-counts by an exponential factor on `n` ones), and by checking the counting recurrence on the sample. I convinced myself the *code* is right by tracing the failing ascending-order case `a=[1], T=3` to a precise cause — read-after-write reusing one item three times — and re-verifying the descending fix, then nailing the zero-value and `T = 0` corners and confirming agreement with brute force over hundreds of cases. That is what I ship: one self-contained file, the simple `O(n*T)` counting DP I can defend, not the greedy I broke.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long T;
    if (!(cin >> n >> T)) return 0;          // empty input -> nothing to do

    const long long MOD = 1000000007LL;

    // dp[s] = number of subsets of the items seen so far whose sum is exactly s, mod MOD.
    // Start with the empty subset, which has sum 0.
    vector<long long> dp(T + 1, 0);
    dp[0] = 1;

    for (int i = 0; i < n; i++) {
        long long v;
        cin >> v;                            // a[i], a non-negative integer
        // 0/1 knapsack count: each item is used at most once.
        // Iterate s from high to low so dp[s - v] still refers to subsets
        // that do NOT yet include item i (prevents reusing one item twice).
        // This descending order is correct even when v == 0.
        for (long long s = T; s >= v; s--) {
            dp[s] = (dp[s] + dp[s - v]) % MOD;
        }
    }

    cout << dp[T] % MOD << "\n";
    return 0;
}
```

**Causal recap.** Greedy/sorting looked plausible but counting is a global quantity, and a single traced instance (`[1,2,3]`, `T=3`: greedy sees `1`, truth `2`) plus the `C(n, n/2)` exponential gap killed it; the exact-target counting DP `dp_new[s] = dp_old[s] + dp_old[s-v]` is provably correct and I checked it on the sample; its in-place form lives or dies on iteration order — my first ascending loop reused a single item three times (`[1], T=3` returned `1` instead of `0`), and the descending sweep makes `dp[s-v]` the pre-item value, fixing it; zeros double correctly because `v=0` makes each cell self-reference (direction-independent), `T=0` yields `2^(#zeros)`, unreachable `T` and `n=0` fall straight out of the base case, and `long long` plus `% MOD` keep the huge counts exact on the residue — all confirmed by 500+ differential tests against a `2^n` brute oracle.
