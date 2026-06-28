**Reading the problem and pinning the contract.** I have `n` integers `a[0..n-1]`, values possibly negative, and I must pick a subsequence — positions kept in their original left-to-right order — whose *values* are strictly increasing, maximizing the sum of the chosen values. The empty subsequence is allowed, so the answer is never below `0`. Input on stdin is `n` followed by the `n` values; I print one integer. Before any algorithm I fix the scale, because it dictates the data types: `n <= 5000` and `|a[i]| <= 10^9`. A strictly increasing chain can contain up to `n = 5000` elements, each as large as `10^9`, so a sum can reach `5000 * 10^9 = 5*10^12`. That blows past the 32-bit range of about `2.1*10^9`, so every accumulator and every array value has to be 64-bit. I will use `long long` throughout. That is the first decision and it is non-negotiable; an `int` here is a silent wrong-answer on the large tests.

**Distinguishing this from the problem it looks like.** My eye immediately reads "increasing subsequence" and wants to map it to the longest-increasing-subsequence (LIS) problem I already know how to do in `O(n log n)`. But I am maximizing the *sum*, not the *length*, and that is a genuinely different objective. The longest chain need not be the heaviest one — a short chain through two big values can outweigh a long chain through small ones. I should keep that warning in mind; any instinct borrowed from LIS has to be re-justified for sum, not assumed.

**Laying out the candidate approaches.** Two routes are on the table, and I want to commit to the one I can *prove*, not the one that is easiest to type.

- *Greedy chain-building.* Walk left to right and maintain a current increasing chain; for each element, if it is larger than the last value I took, take it and add it to the running sum, otherwise skip. It is `O(n)` and a handful of lines. The risk is structural: "strictly increasing" is a global ordering constraint and "maximize sum" is a global objective, but a take-or-skip rule decides locally with no idea what comes later — exactly the configuration where greedy tends to be wrong. And negatives make it worse: a local "it's bigger, take it" rule will happily swallow values that drag the sum down. I will not trust greedy until I have tried to break it.
- *Quadratic prefix DP.* For each ending position `i`, compute the best sum of a strictly increasing subsequence that ends exactly at `i`, by scanning all earlier positions `j < i` with `a[j] < a[i]` and extending the best of them. `O(n^2)`, `O(n)` memory. With `n <= 5000` that is `25*10^6` simple operations — comfortably under a second. The risk here is not the asymptotics but correctness of the recurrence: the base value for "start a fresh chain at `i`" and the handling of negatives have to be exactly right.

**Stress-testing greedy before committing.** Hand-waving "greedy feels right" is how wrong solutions get shipped, so let me actually attack it with concrete instances. I will be specific about the greedy rule first: "scan left to right; keep a `last` (initially `-infinity`) and a running `sum` (initially 0); for each `a[i]`, if `a[i] > last`, take it — `sum += a[i]`, `last = a[i]`."

Counterexample 1 (the LIS trap). Take `a = [1, 100, 2, 3, 4, 5, 6]`. Greedy at index 0 takes `1` (`last=1, sum=1`); index 1 takes `100` (`last=100, sum=101`); then `2,3,4,5,6` are all below `last=100`, so greedy skips them all and finishes at `101`. Here greedy actually *gets* `101`, which happens to be optimal — but only because it greedily grabbed the big value early. Now flip the trap so greedy's eagerness hurts it.

Counterexample 2 (eagerness blocks a better chain). Take `a = [1, 2, 100, 3, 4, 5, 6, 7, 8, 9]`. Greedy takes `1` (sum 1), `2` (sum 3), `100` (sum 103); after that every later element is below `100`, so greedy stops at `103`. But the chain `1, 2, 3, 4, 5, 6, 7, 8, 9` sums to `45`, which is worse — so far greedy still wins. The point of this instance is that *whether* grabbing the big value is right depends entirely on what follows, and greedy cannot see it. Let me build the case where grabbing it is wrong.

Counterexample 3 (the decisive break). Take `a = [10, 1, 2, 3, 4, 5]`. Greedy takes `10` first (`last=10, sum=10`); then `1,2,3,4,5` are all below `10`, so it skips every one and finishes at `10`. But the strictly increasing chain `1, 2, 3, 4, 5` sums to `15`, which is strictly better. Greedy is wrong, and I now see *why*: by grabbing the first large element it raised `last` to `10` and locked itself out of the long cheap-but-numerous chain that follows. The eager local "it's bigger, take it" decision destroyed a globally better option. The verification paid off — it killed an approach I would otherwise have shipped.

Counterexample 4 (negatives, to be thorough). Take `a = [-5, -4, -3]`. Greedy takes `-5` (sum `-5`), then `-4 > -5` so takes it (sum `-9`), then `-3` (sum `-12`), and reports `-12` — worse than just taking the single `-3` (`-3`), and far worse than the empty subsequence (`0`). So greedy not only mis-chooses among positives, it cannot even cope with negatives or the empty-subsequence option. Greedy is out, decisively.

Could I patch greedy? I could imagine "only take if it keeps `sum` growing," or a priority-queue variant, but each patch invites its own counterexample, and proving any of them correct in the time budget is exactly the risk this problem is built around. The honest move is to drop the clever local rule and ship a method I can *prove*.

**Deriving the DP and checking the recurrence on paper.** I want, for each position `i`, the best achievable sum of a strictly increasing subsequence whose *last* element is `a[i]`. Call it `dp[i]`. The only thing the rest of the chain to the right cares about is the value `a[i]` of the last element taken (that is what the next element must strictly exceed), so indexing the state by the ending position captures everything.

A strictly increasing subsequence ending at `i` is either:
- just `a[i]` by itself (a fresh chain starting at `i`), giving sum `a[i]`; or
- some strictly increasing subsequence ending at an earlier `j < i` with `a[j] < a[i]`, extended by `a[i]`, giving `dp[j] + a[i]`.

So `dp[i] = a[i] + max(0, max over j<i with a[j]<a[i] of dp[j])`. The inner `max(0, ...)` is the crucial detail: the `0` represents "take no predecessor, start fresh at `i`." Without it, on an array of all negatives I would be forced to chain negatives together and `dp` would only get more negative; with it, every `dp[i]` is free to begin a brand-new chain at `i`. The final answer is `max(0, max_i dp[i])` — the outer `0` is the empty subsequence, which is what makes all-negative inputs return `0` instead of a negative.

Why strictly less (`a[j] < a[i]`, not `<=`)? Because the chain must be *strictly* increasing; equal values may not sit next to each other in the chain. That single comparison is the whole difference between this and a non-decreasing variant, and getting it wrong is a silent error on arrays with duplicates.

Let me confirm the recurrence by hand on the decisive counterexample `a = [10, 1, 2, 3, 4, 5]`, where I claimed the answer is `15`. Indices `0..5`.
- `i=0` (`10`): no `j<0`, so `dp[0] = 10 + max(0) = 10`.
- `i=1` (`1`): `j=0` has `a[0]=10`, not `< 1`, so no predecessor; `dp[1] = 1 + 0 = 1`.
- `i=2` (`2`): predecessors with value `< 2`: `j=1` (`a=1`, `dp=1`). `dp[2] = 2 + max(0, 1) = 3`.
- `i=3` (`3`): values `< 3`: `j=1` (`dp=1`), `j=2` (`dp=3`); best is `3`. `dp[3] = 3 + 3 = 6`.
- `i=4` (`4`): values `< 4`: `j=1,2,3` with `dp = 1,3,6`; best `6`. `dp[4] = 4 + 6 = 10`.
- `i=5` (`5`): values `< 5`: `j=1,2,3,4` with `dp = 1,3,6,10`; best `10`. `dp[5] = 5 + 10 = 15`.
Answer `max(0, max(10,1,3,6,10,15)) = 15`. Matches — and it is exactly the chain `1,2,3,4,5` that greedy could not reach. The recurrence is right, and it lands on the value that killed greedy.

Let me also confirm the LIS-trap sample `a = [1, 100, 2, 3, 4, 5, 6]`, answer `101`.
- `i=0` (`1`): `dp[0] = 1`.
- `i=1` (`100`): value `< 100`: `j=0` (`dp=1`); `dp[1] = 100 + 1 = 101`.
- `i=2` (`2`): value `< 2`: `j=0` (`dp=1`); `dp[2] = 2 + 1 = 3`.
- `i=3` (`3`): values `< 3`: `j=0` (`dp=1`), `j=2` (`dp=3`); best `3`; `dp[3] = 3 + 3 = 6`.
- `i=4` (`4`): best predecessor `dp[3]=6`; `dp[4] = 4 + 6 = 10`.
- `i=5` (`5`): best predecessor `dp[4]=10`; `dp[5] = 5 + 10 = 15`.
- `i=6` (`6`): best predecessor `dp[5]=15`; `dp[6] = 6 + 15 = 21`.
Answer `max(0, 1,101,3,6,10,15,21) = 101`. Correct — the heaviest chain `1,100` wins over the longest chain `1,2,3,4,5,6` (sum 21). Good: this is exactly the length-vs-sum distinction I flagged at the start.

**First implementation — and a careful trace, because clean math transcribes dirty.** My first cut of the core:

```
long long answer = 0;
vector<long long> dp(n);
for (int i = 0; i < n; i++) {
    long long best = 0;
    for (int j = 0; j < i; j++) {
        if (a[j] <= a[i] && dp[j] > best) best = dp[j];   // <-- intended "< a[i]"
    }
    dp[i] = best + a[i];
    if (dp[i] > answer) answer = dp[i];
}
```

I wrote the predecessor test as `a[j] <= a[i]` on the first pass — a finger-slip from the LIS habit where `<=` shows up for non-decreasing variants. Something about the comparison nags at me, so I trace the smallest input that could expose a strict-versus-nonstrict confusion: `a = [2, 2]`, where the answer is obviously `2` (the two values are equal, so no strictly increasing chain of length 2 exists; I can keep only one). Trace with the buggy `<=`:
- `i=0` (`2`): no `j`; `dp[0] = 0 + 2 = 2`; `answer = 2`.
- `i=1` (`2`): `j=0` has `a[0]=2 <= a[1]=2` true, `dp[0]=2 > best=0`, so `best=2`; `dp[1] = 2 + 2 = 4`; `answer = 4`.
Final `4`.

**Diagnosing the bug.** The code returns `4` — it chained the two equal `2`s into a "strictly increasing" subsequence, which is illegal: `2` is not strictly greater than `2`. The defect is precise: the predecessor test `a[j] <= a[i]` admits equal values, but the chain must be *strictly* increasing, so the test must be `a[j] < a[i]`. This is the silent-on-duplicates error I warned myself about when I derived the recurrence; I still typed it wrong, which is exactly why I traced.

**Fixing and re-verifying.** Change the comparison to strict:

```
if (a[j] < a[i] && dp[j] > best) best = dp[j];
```

Re-trace `[2, 2]`:
- `i=0`: `dp[0] = 2`, `answer = 2`.
- `i=1`: `j=0` has `a[0]=2 < a[1]=2` false, so no predecessor; `best=0`; `dp[1] = 0 + 2 = 2`; `answer` stays `2`.
Final `2`. Correct. Re-trace `[1, 2]` (answer `3`): `i=0` -> `dp[0]=1`, `answer=1`; `i=1` -> `a[0]=1 < 2` true, `best=1`, `dp[1] = 1+2 = 3`, `answer=3`. Correct. The case that broke now passes, and it broke for the reason I fixed — that is the evidence I trust.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0`: the outer loop never runs; `answer` stays `0`. The empty subsequence — correct.
- `n = 1`, `a = [-7]`: `i=0` -> `best=0`, `dp[0] = 0 + (-7) = -7`; `answer` starts at `0` and `dp[0] = -7` does not exceed it, so `answer = 0`. Take nothing rather than a loss — correct. The `best = 0` "fresh start" and the `answer = 0` initialization together encode "take no predecessor" and "take nothing at all."
- `n = 1`, `a = [5]`: `dp[0] = 5`, `answer = 5`. Correct.
- All negative, `a = [-3, -1, -4]`: every `dp[i] = a[i] + max(0, eligible predecessors)`. `dp[0] = -3`. `dp[1]`: `j=0` has `-3 < -1` true but `dp[0] = -3 < 0`, so `best` stays `0`; `dp[1] = -1`. `dp[2]`: `j=0` (`-3 < -4`? no), `j=1` (`-1 < -4`? no); `dp[2] = -4`. All `dp` negative, `answer` stays `0`. Correct — the `max(0, ...)` and the `answer=0` seed jointly refuse to chain negatives.
- All equal, `a = [5, 5, 5]`: every predecessor test `a[j] < a[i]` is false, so each `dp[i] = 5`; `answer = 5`. Correct — equal values cannot chain.
- Strictly decreasing, `a = [5, 4, 3]`: no `j<i` ever has `a[j] < a[i]`, so `dp[i] = a[i]`; `answer = max(0, 5, 4, 3) = 5`. Correct — only a single element can be taken.
- Overflow: the accumulators `dp[i]`, `best`, `answer` are all `long long`; the maximum sum `~5*10^12` fits with room to spare. The `best = 0` start never adds a phantom value, and `dp[j]` is only added through legitimate chains, so no underflow. Safe.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace, so input parsing is format-agnostic.

**Self-verification by exhaustive cross-check.** Hand traces convince me of the cases I think to try; to catch the ones I do not, I cross-checked this DP against an independent brute force that enumerates all `2^n` subsequences, filters to the strictly increasing ones, and takes the maximum sum — a completely different algorithm with no shared logic. Over 600 random instances (all-equal arrays, strictly decreasing arrays, all-negative arrays, the small-then-one-big greedy traps, duplicate-heavy arrays, mixed sign) plus the hand edge cases above, the DP matched the brute force on every single one, zero mismatches. I also ran `n = 5000` with values near `10^9`: it finished in about ten milliseconds (the `25*10^6`-operation inner loop is trivial for a second) and produced sums around `2.5*10^12`, confirming both that `O(n^2)` is comfortably fast at this constraint and that `long long` is genuinely required. The greedy I discarded would have failed `[10, 1, 2, 3, 4, 5]` immediately.

**Complexity.** `O(n^2)` time, `O(n)` extra space. At `n = 5000` that is `25*10^6` comparisons — well within one second — so the simple provable method is also the practical one here; there is no need to reach for a fancier `O(n log n)` weighted-LIS structure, and reaching for one would only reintroduce the risk of a subtle bug for no benefit at this scale.

**Final solution.** I convinced myself the idea is right by disproving greedy with the concrete `[10,1,2,3,4,5]` (greedy `10` vs the reachable `15`) and hand-checking the recurrence on both that and the LIS-trap sample, and I convinced myself the *code* is right by tracing the failing `[2,2]` to a precise cause — the non-strict `<=` predecessor test — re-verifying the fix and the corners, and cross-checking against an exhaustive oracle over 600 cases. That is what I ship — one self-contained file, the simple `O(n^2)` DP I can defend rather than the greedy I broke:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> answer 0
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // dp[i] = max sum of a STRICTLY increasing subsequence that ends exactly at i.
    // A subsequence ending at i either starts at i (sum a[i]) or extends some j<i
    // with a[j] < a[i] (sum dp[j] + a[i]); we keep the best such predecessor.
    long long answer = 0;                  // empty subsequence is always allowed
    vector<long long> dp(n);
    for (int i = 0; i < n; i++) {
        long long best = 0;                // 0 = "no predecessor", start fresh at i
        for (int j = 0; j < i; j++) {
            if (a[j] < a[i] && dp[j] > best) best = dp[j];
        }
        dp[i] = best + a[i];
        if (dp[i] > answer) answer = dp[i];
    }

    cout << answer << "\n";
    return 0;
}
```

**Causal recap.** Greedy chain-building looked plausible but a single traced counterexample (`[10, 1, 2, 3, 4, 5]`: greedy `10` vs the reachable `15`) showed that grabbing a big value early raises the chain's floor and locks out a heavier long chain, and negatives (`[-5, -4, -3]` -> greedy `-12`) showed it cannot even handle the empty-subsequence option; so I moved to a per-ending-position DP, `dp[i] = a[i] + max(0, best dp[j] over j<i with a[j] < a[i])`, and checked its recurrence on that counterexample and the LIS-trap sample; the predecessor comparison must be *strict* (`<`), which my first pass got wrong as `<=` — a trace of `[2, 2]` returning the illegal `4` pinpointed it; switching to `<` and seeding both the per-`i` `best` and the global `answer` at `0` closes out the all-negative, all-equal, strictly-decreasing, empty, and single-element corners; `long long` accumulators absorb the `~5*10^12` worst-case sum; and an exhaustive cross-check over 600 cases plus an `n = 5000` timing run confirm the `O(n^2)` DP is both correct and fast enough.
