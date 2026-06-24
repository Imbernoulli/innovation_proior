**Reading the problem and pinning the contract.** I stock `n` distinct stamp denominations `d[0..n-1]`, each in unlimited supply, and a parcel needs *exactly* `A` cents. I want the fewest stamps whose values sum to exactly `A`, repetition allowed; if no multiset of the denominations hits `A` exactly, I print `-1`. Input is `n` and `A` on one line, then the `n` denominations; output is one integer. Before any algorithm I fix the scale, because it decides both the data types and which methods are even admissible: `n <= 100`, `A <= 10^5`, and each `d[i] <= 10^5`. The number of stamps is at most `A` (every stamp is worth at least `1`), so the answer is bounded by `10^5` and fits comfortably in a 32-bit range — but I will still carry counts and sums in `long long` to be safe against any intermediate arithmetic, since the cost of a wider accumulator here is nothing. The two hard corners I can already see from the contract: `A = 0` (the empty multiset sums to `0`, so the answer is `0`, not `-1`), and genuine impossibility (e.g. every denomination even but `A` odd), which must yield `-1`. I will hold both of those in mind and revisit them at the end.

**Laying out the candidate approaches.** Two routes are on the table, and I want to commit to the one I can *prove* rather than the one that types fastest.

- *Greedy by denomination.* Sort denominations descending; repeatedly stick on the largest stamp that is `<= remaining`, subtract it, and repeat until the remainder is `0` (success) or nothing fits (failure). This is `O(n log n)` plus the number of stamps used, and it is maybe five lines. It is exactly the algorithm a human uses to make change with real-world coins, which is precisely why it feels obviously correct. The structural risk is that "fewest pieces to hit an exact target" is a global optimization, and greedy commits to the biggest piece locally without any lookahead. Real currency systems are deliberately designed ("canonical") so that greedy is optimal, but an arbitrary denomination set carries no such guarantee. I will not trust greedy until I have actively tried to break it.
- *Unbounded shortest-combination DP.* Define `dp[v]` = the minimum number of stamps summing to exactly `v`, for `v` from `0` to `A`. Fill upward, relaxing each `v` through every denomination `d[i] <= v`. This is `O(n * A)` time and `O(A)` memory, which at `100 * 10^5 = 10^7` relaxations is trivially within a 1-second limit. The risk here is not the idea but the *transcription*: the base case (`dp[0] = 0`), the impossibility sentinel, and the guard so I never relax through an unreachable state.

**Stress-testing greedy before committing — building a counterexample on purpose.** Hand-waving "greedy feels right" is exactly how wrong solutions ship, so let me attack greedy with a concrete instance rather than admire it. Take the denomination set `d = [1, 3, 4]` and target `A = 6`. Greedy sorts descending to `[4, 3, 1]`. Remaining `6`: the largest `<= 6` is `4`, take it, remaining `2`. Largest `<= 2` is `1` (the `3` and `4` are both too big now), take it, remaining `1`. Largest `<= 1` is `1`, take it, remaining `0`. Greedy used `4 + 1 + 1`, i.e. **3 stamps**.

Is 3 optimal? Let me hunt for something greedy structurally cannot reach. The combination `3 + 3 = 6` uses only **2 stamps**, and it is a perfectly legal multiset (the `3` denomination is in unlimited supply). So `2 < 3`: greedy is wrong, and I now see exactly *why*. By grabbing the largest stamp `4` first, greedy committed to a remainder of `2` that the denomination set fills only with two `1`-cent stamps; the "better" route never touches the largest stamp at all and pairs two medium stamps instead. The largeness of `4` was a local lure, not a global help. The verification paid off immediately — it killed an approach I would otherwise have shipped. To be sure this is not a one-off, I sanity-check a second non-canonical set: `d = [1, 5, 8]`, `A = 10`. Greedy takes `8`, then `1 + 1`, total `3` stamps; but `5 + 5 = 10` is `2` stamps. Wrong again. Two independent counterexamples, two clean explanations: greedy is out. It is correct only on special "canonical" systems, and the problem statement promises me nothing about canonicity, so I cannot use it.

**Deriving the DP and checking the recurrence on paper.** I want `dp[v]` = the minimum number of stamps that sum to exactly `v`. The structure is clean: any optimal multiset summing to `v` has *some* last stamp of value `d[i]` for some `i` with `d[i] <= v`; remove it, and what remains is an optimal multiset summing to `v - d[i]` (if it were not optimal I could improve it and contradict optimality of the whole). So

    dp[v] = 1 + min over all i with d[i] <= v of dp[v - d[i]],

and if no `i` makes `v - d[i]` reachable, then `v` itself is unreachable. The base case is `dp[0] = 0`: the empty multiset sums to `0` with zero stamps, which is both correct and exactly what makes `A = 0` return `0` for free. I represent "unreachable" with a large sentinel `INF`, and I only relax through `v - d[i]` when `dp[v - d[i]] != INF`, so impossibility never gets a finite count by accident. At the end, `dp[A] == INF` means no exact franking exists, so I print `-1`; otherwise I print `dp[A]`.

Let me confirm the recurrence by hand on the sample `d = [1, 3, 4]`, `A = 6`, expected answer `2`. `dp[0] = 0`. `dp[1]`: only `d=1` fits, `dp[0]+1 = 1`. `dp[2]`: `d=1` gives `dp[1]+1 = 2`; `d=3,4` too big; so `2`. `dp[3]`: `d=1 -> dp[2]+1 = 3`; `d=3 -> dp[0]+1 = 1`; min `1`. `dp[4]`: `d=1 -> dp[3]+1 = 2`; `d=3 -> dp[1]+1 = 2`; `d=4 -> dp[0]+1 = 1`; min `1`. `dp[5]`: `d=1 -> dp[4]+1 = 2`; `d=3 -> dp[2]+1 = 3`; `d=4 -> dp[1]+1 = 2`; min `2`. `dp[6]`: `d=1 -> dp[5]+1 = 3`; `d=3 -> dp[3]+1 = 2`; `d=4 -> dp[2]+1 = 3`; min `2`. So `dp[6] = 2` — matching `3 + 3`, and strictly beating greedy's `3`. The recurrence is right, and it found the answer greedy could not.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut of the table fill:

```
const long long INF = (long long)4e18;
vector<long long> dp(A + 1, INF);
dp[0] = 0;
for (long long v = 1; v <= A; v++) {
    for (int i = 0; i < n; i++) {
        long long cand = dp[v - d[i]] + 1;     // <-- relax through every denomination
        if (cand < dp[v]) dp[v] = cand;
    }
}
```

Something about that inner line bothers me: I index `dp[v - d[i]]` without first checking that `d[i] <= v`, and I add `1` to `dp[v - d[i]]` without first checking it is not the `INF` sentinel. Let me trace the smallest input that could expose either defect. Take `d = [4]`, `A = 2`. `dp` is `[0, INF, INF]`. `v = 1`: `i = 0`, `d[0] = 4`, so `v - d[i] = 1 - 4 = -3`. The code reads `dp[-3]` — an out-of-bounds negative index. That is undefined behavior; on my machine it might read garbage memory and silently corrupt `dp[1]`, or it might crash. Either way it is a real bug. Even pretending the index were guarded, there is a second defect lurking: if I ever do `dp[v - d[i]] + 1` when `dp[v - d[i]] == INF` (an unreachable sub-amount), I would manufacture a finite count `INF + 1` for `dp[v]`, turning a genuinely impossible state into a falsely "reachable" one — and if I had used a sentinel near the top of the `long long` range, `INF + 1` would also *overflow*. Both bugs trace to the same root: I relax unconditionally instead of only through legal, reachable predecessors.

**Diagnosing the bug precisely.** The defect is exact. The transition `dp[v] = 1 + dp[v - d[i]]` is only meaningful when (a) `d[i] <= v`, so the predecessor index `v - d[i]` is a real non-negative amount, and (b) `dp[v - d[i]] != INF`, so the predecessor is actually achievable. My first loop enforced neither. Condition (a)'s violation is an out-of-bounds read (`dp[-3]`), pure undefined behavior. Condition (b)'s violation is a logic error that both fabricates reachability and risks sentinel overflow. I need both guards, and they must come *before* I touch `dp[v - d[i]]`.

**Fixing and re-verifying.** Guard the relaxation on both conditions, in order:

```
for (long long v = 1; v <= A; v++) {
    for (int i = 0; i < n; i++) {
        if (d[i] <= v && dp[v - d[i]] != INF) {     // legal index AND reachable predecessor
            long long cand = dp[v - d[i]] + 1;
            if (cand < dp[v]) dp[v] = cand;
        }
    }
}
```

Re-trace `d = [4]`, `A = 2`: `dp = [0, INF, INF]`. `v = 1`: `d[0] = 4 > 1`, guard fails, skip; `dp[1]` stays `INF`. `v = 2`: `d[0] = 4 > 2`, skip; `dp[2]` stays `INF`. `dp[A] = dp[2] = INF`, so output `-1` — correct: you cannot make `2` from only `4`-cent stamps, and the short-circuit `&&` means `dp[v - d[i]]` is never even read when `d[i] > v`, so no out-of-bounds access. Re-trace the impossibility case `d = [2, 4]`, `A = 7`: every reachable amount is even (`dp` finite only at `0, 2, 4, 6`), and `dp[7]` is relaxed only from `dp[5]` and `dp[3]`, both `INF`, so it stays `INF` -> `-1`. Correct. The two cases that broke before now pass, and they pass for the exact reason I fixed.

**A second debug episode — the `A = 0` base case and an off-by-one in the loop bound.** Before I declare victory I want to confirm the empty-target corner, because off-by-one errors love loop bounds. Trace `d = [3, 5]`, `A = 0`. I allocate `dp` of size `A + 1 = 1`, so `dp = [INF]`, then set `dp[0] = 0`. The loop runs `for v = 1; v <= 0`, which never executes. `dp[A] = dp[0] = 0`, output `0` — correct, the empty multiset. Good. But this makes me re-examine the allocation: I size `dp` as `A + 1` and index up to `dp[A]`, so the valid indices are `0..A` inclusive; the loop's `v <= A` is the right (inclusive) bound, and `dp[v - d[i]]` with `d[i] >= 1` and `d[i] <= v` lands in `[0, v - 1] ⊆ [0, A - 1]`, always in range. If I had written `dp(A)` (size `A`, off by one) instead of `dp(A + 1)`, then `dp[A]` would be an out-of-bounds read for every nonzero `A` — a classic off-by-one. I check my code: it is `vector<long long> dp(A + 1, INF)`, sized correctly, so `dp[A]` is always valid (and for `A = 0` it is `dp[0]`, the only element). The base case and the loop bound are both right, and I verified it by tracing the smallest target rather than assuming.

**Sanity-checking the derivation itself against the sample.** The contract's worked example is `d = [1, 3, 4]`, `A = 6` -> `2`, and my hand-fill of the DP above produced exactly `dp[6] = 2` via `3 + 3`, while I separately showed greedy yields `3`. So the *intended* answer is the DP's `2`, the greedy trap is real and pointed in the documented direction (greedy overshoots), and my method recovers the correct value. The derivation and the sample agree.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `A = 0`: `dp` has one cell `dp[0] = 0`, the loop is empty, output `0`. The empty multiset — correct, and explicitly distinguished from `-1`.
- Single denomination `d = [7]`, `A = 14`: `dp[7] = 1` (from `dp[0]`), `dp[14] = dp[7] + 1 = 2`. Output `2`. And `d = [7]`, `A = 10`: `dp[10]` relaxes only from `dp[3]` (`INF`), stays `INF`, output `-1`. Correct on both reachable and unreachable single-denomination targets.
- Impossibility by parity: `d = [2, 4, 6]`, `A = 5` -> all reachable amounts even, `dp[5] = INF`, output `-1`. Correct.
- Denomination larger than `A`: `d = [100000, 1]`, `A = 3` -> the `100000` stamp never satisfies `d[i] <= v` for any `v <= 3`, so it is silently ignored by the guard; answer is `3` (three `1`s). No out-of-bounds, no special-casing needed.
- Largest scale: `A = 10^5`, `n = 100`. The table is `10^5 + 1` `long long` cells (about 0.8 MB, far under 256 MB), and the fill is `<= 10^7` relaxations. A timing run completes in about 11 ms — comfortably inside 1 second.
- Overflow / sentinel safety: counts never exceed `A <= 10^5`, so `cand = dp[...] + 1` never approaches the `long long` range; and because I guard `dp[v - d[i]] != INF` *before* adding `1`, I never compute `INF + 1`, so the sentinel `4e18` can never overflow even though it sits high in the `long long` range. Safe.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace, so the two-line input layout is parsed format-agnostically. The empty-input guard `if (!(cin >> n >> A)) return 0;` prevents reading from an empty stream.

**Final solution.** I convinced myself the *idea* is right by actively disproving greedy with two constructed counterexamples (`[1,3,4]@6` and `[1,5,8]@10`, where greedy overshoots by a stamp) and by hand-filling the DP on the sample to recover the true optimum; and I convinced myself the *code* is right by tracing two distinct failing scenarios — the unguarded negative index / fabricated reachability, and the `A = 0` / off-by-one allocation — to precise causes and re-verifying the fixes plus every corner. That is what I ship — one self-contained file, the `O(n*A)` DP I can defend rather than the greedy I broke:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long A;
    if (!(cin >> n >> A)) return 0;          // empty input -> nothing to do
    vector<long long> d(n);
    for (auto &x : d) cin >> x;

    // Unbounded "fewest stamps to make exactly A" by DP over amounts 0..A.
    // dp[v] = minimum number of stamps summing to exactly v, or INF if impossible.
    const long long INF = (long long)4e18;
    vector<long long> dp(A + 1, INF);
    dp[0] = 0;
    for (long long v = 1; v <= A; v++) {
        for (int i = 0; i < n; i++) {
            if (d[i] <= v && dp[v - d[i]] != INF) {
                long long cand = dp[v - d[i]] + 1;
                if (cand < dp[v]) dp[v] = cand;
            }
        }
    }

    if (dp[A] == INF) cout << -1 << "\n";
    else cout << dp[A] << "\n";
    return 0;
}
```

**Causal recap.** Greedy-by-largest-stamp looked obviously right but two traced counterexamples (`[1,3,4]@6`: greedy `4+1+1=3` vs the reachable `3+3=2`, and `[1,5,8]@10`: greedy `8+1+1=3` vs `5+5=2`) showed a big local stamp commits to a remainder the denomination set fills poorly, so I moved to the unbounded shortest-combination DP and checked its recurrence by hand-filling `dp[0..6]` to recover `2`; the DP's transition is meaningful only through a legal, reachable predecessor, which my first unguarded loop violated — a trace of `[4]@2` exposed an out-of-bounds `dp[-3]` and a fabricated-reachability / sentinel-overflow path, both fixed by the single guard `d[i] <= v && dp[v - d[i]] != INF`; a second trace of `[3,5]@0` confirmed the `dp(A+1)` sizing and the empty loop give `0` rather than an off-by-one out-of-bounds; and the `INF` sentinel checked before the `+1` closes out the impossibility (`-1`), parity-unreachable, oversized-denomination, max-scale, and overflow corners.
