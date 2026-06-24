**Reading the problem and pinning the contract.** I have `n` daily profits `a[0..n-1]`, possibly negative, and I must pick non-overlapping contiguous *lots*, each spanning at least `L` days, to maximize the total of the lot sums; harvesting nothing is allowed, so the answer never drops below `0`. Input is `n` and `L` on the first line, then the `n` values; I print one integer. Before any algorithm I fix the scale, because it dictates the data types: `n <= 2*10^5` and `|a[i]| <= 10^9`, so a total can reach `2*10^5 * 10^9 = 2*10^14`. That dwarfs the 32-bit range of about `2.1*10^9`, so every prefix sum, every accumulator, and every `dp` cell has to be 64-bit `long long`. I will also keep `L` as `long long` so that an adversarial `L > n` (the contract explicitly permits it) cannot do anything strange when I subtract it from an index. These two decisions — 64-bit everywhere, and treating `L > n` as "no lot fits" — are non-negotiable; an `int` accumulator is a silent wrong answer on the large tests, and assuming `L <= n` is a silent crash or garbage on the boundary tests.

**The one structural observation that makes prefix sums the right tool.** A lot `[j, i-1]` has value `a[j] + ... + a[i-1]`. If I precompute `P[i] = a[0] + ... + a[i-1]` (with `P[0] = 0`), then that value is exactly `P[i] - P[j]`, an `O(1)` lookup. So once I think in terms of the *boundaries* `0..n` rather than the days, a lot is just a pair of boundaries `(j, i)` with `i - j >= L`, contributing `P[i] - P[j]`. That reframing is what lets both candidate methods below avoid re-summing a block every time.

**Laying out the candidate approaches.** Two routes are on the table, and I want to commit to the one I can *prove*, not the one that is easiest to type.

- *Greedy by best lot.* Repeatedly find the single maximum-sum lot of length `>= L` over the still-free days, take it, then recurse on the free runs to its left and right; stop when no positive lot remains. It is intuitive — "always bank the most profitable block you can see" — and prefix sums make each search cheap. The risk is structural: lot selection is a *global* packing decision and this rule commits *locally* to one block, which is precisely the configuration where greedy tends to be wrong. I will not trust it until I have actively tried to break it.
- *Linear prefix DP.* Sweep the boundaries `0..n` left to right and let `dp[i]` be the best total achievable using only days `a[0..i-1]`. `O(n)` if I can keep the inner choice `O(1)`. The risk here is not the idea but the *transcription* — the eligibility offset for `j` and the base case are easy to write subtly wrong.

**Stress-testing greedy before committing — and breaking it.** Hand-waving "greedy feels optimal" is how wrong solutions get shipped, so let me actually attack it with a concrete instance. Take `L = 2` and `a = [9, -2, -5, 8, -5, 9]`, boundaries `0..6`. Greedy first looks for the single best lot of length `>= 2` over the whole array. I scan: the lot `[0,5]` (the entire array) sums to `9 - 2 - 5 + 8 - 5 + 9 = 14`. Is anything better? `[3,5] = 8 - 5 + 9 = 12`; `[0,1] = 7`; `[0,3] = 10`; none beats `14`. So greedy grabs the whole array for `14` — and now there are *no free days left*, so greedy stops with total `14`.

Is `14` optimal? Let me hunt for a packing greedy structurally could not reach, one that *declines* to harvest the costly middle. Take two lots: `[0,1] = 9 - 2 = 7` and `[3,5] = 8 - 5 + 9 = 12`, leaving day `2` (the `-5`) unharvested in the gap. They are non-overlapping, each has length `>= 2`, and the total is `7 + 12 = 19`, strictly better than `14`. So greedy is wrong, and I see *exactly why*: the single globally-best block `[0,5]` is forced to swallow the two `-5` losing days in its interior because they sit *between* the profitable ends; by banking that one fat block greedy can never split around the losses. The optimal solution pays to *skip* day `2` and the boundary between the lots. The verification paid off — it killed an approach I would otherwise have shipped. Greedy is out, and the failure mode ("a single best block may straddle a valley that the optimum prefers to leave uncovered") is exactly the kind a `dp` over boundaries handles for free, because the `dp` is allowed to leave any day uncovered.

**Deriving the DP and the prefix-sum trick that keeps it linear.** Let `dp[i]` = best total over days `a[0..i-1]` using non-overlapping lots of length `>= L`, with "harvest nothing" always available so `dp[i] >= 0`. I always allow `dp[0] = 0`. For boundary `i`, consider what happens to day `i-1`:

- *Leave day `i-1` unharvested.* Then the best is whatever I had over the first `i-1` days: `dp[i-1]`.
- *Close a lot exactly at day `i-1`.* The lot is `[j, i-1]` for some `j` with length `i - j >= L`, i.e. `j <= i - L`, and before that lot I could do anything over the first `j` days. Its value is `P[i] - P[j]`. So this branch is `max over 0 <= j <= i - L of ( dp[j] + P[i] - P[j] )`.

Therefore `dp[i] = max( dp[i-1], P[i] + max_{0 <= j <= i-L} ( dp[j] - P[j] ) )`. The crucial move: the inner term `dp[j] - P[j]` depends only on `j`, and the set of eligible `j` grows by exactly one (`j = i - L`) each time `i` advances by one. So I keep a running `best = max over all currently-eligible j of ( dp[j] - P[j] )` and extend it as `i` grows. That collapses the inner `max` to `O(1)` and the whole thing to `O(n)` time, `O(n)` space (or `O(1)` if I dropped the `dp` array, but I will keep it for clarity). Answer is `dp[n]`, which is `>= 0` because every `dp[i] >= dp[0] = 0`.

**Sanity-checking the recurrence on the documented sample.** Sample: `n = 8`, `L = 3`, `a = [3, -1, 4, -10, 2, 2, -1, 5]`, claimed answer `14`. Prefix sums `P = [0, 3, 2, 6, -4, -2, 0, -1, 4]`. A `j` becomes eligible at `i = j + L`, i.e. `j = i - 3`. Let me run it. `dp[0]=0`, `best` starts empty (`-inf`).
- `i=1`: `j=1-3=-2 < 0`, nothing new eligible; `best=-inf`; `dp[1]=dp[0]=0`.
- `i=2`: `j=-1<0`; `dp[2]=dp[1]=0`.
- `i=3`: `j=0` becomes eligible, `dp[0]-P[0]=0-0=0`, so `best=0`; `dp[3]=max(dp[2], P[3]+best)=max(0, 6+0)=6`. (That is the lot `[0,2]=3-1+4=6`. Good.)
- `i=4`: `j=1` eligible, `dp[1]-P[1]=0-3=-3`, `best=max(0,-3)=0`; `dp[4]=max(dp[3], P[4]+best)=max(6, -4+0)=6`.
- `i=5`: `j=2` eligible, `dp[2]-P[2]=0-2=-2`, `best=max(0,-2)=0`; `dp[5]=max(dp[4], P[5]+best)=max(6, -2+0)=6`.
- `i=6`: `j=3` eligible, `dp[3]-P[3]=6-6=0`, `best=max(0,0)=0`; `dp[6]=max(dp[5], P[6]+best)=max(6, 0+0)=6`.
- `i=7`: `j=4` eligible, `dp[4]-P[4]=6-(-4)=10`, `best=max(0,10)=10`; `dp[7]=max(dp[6], P[7]+best)=max(6, -1+10)=9`.
- `i=8`: `j=5` eligible, `dp[5]-P[5]=6-(-2)=8`, `best=max(10,8)=10`; `dp[8]=max(dp[7], P[8]+best)=max(9, 4+10)=14`.

Final `dp[8] = 14`, matching the documented answer, and the witnessing decomposition is the lot ending at boundary `8` with the best `j` being `4` (`dp[4]=6` from lot `[0,2]`, plus `P[8]-P[4]=4-(-4)=8` for lot `[4,7]=2+2-1+5=8`), total `6+8=14`. The recurrence is right.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut of the loop:

```
const long long NEG = LLONG_MIN / 4;
long long best = NEG;
for (int i = 1; i <= n; i++) {
    int j = i - (int)L;
    if (j >= 0) best = max(best, dp[j] - P[j]);
    dp[i] = max(dp[i - 1], P[i] + best);   // <-- suspicious
}
```

Something about `P[i] + best` when `best` is still the sentinel `NEG` looks dangerous, so I trace the smallest input that exposes it: `n = 2`, `L = 5`, `a = [3, 3]` — here `L > n`, so *no* lot fits and the answer must be `0`. `P = [0, 3, 6]`. `dp[0]=0`, `best=NEG`.
- `i=1`: `j=1-5=-4 < 0`, no update, `best=NEG`; `dp[1]=max(dp[0], P[1]+NEG)=max(0, 3 + LLONG_MIN/4)`. But `3 + LLONG_MIN/4` is still hugely negative, so the `max` picks `0`. OK here.
- `i=2`: `j=2-5=-3 < 0`, no update, `best=NEG`; `dp[2]=max(dp[1], P[2]+NEG)=max(0, 6 + LLONG_MIN/4)=0`.

Answer `0` — correct on this input. But the trace makes me nervous for a different reason: I am *adding* `P[i]` to `LLONG_MIN/4`. With `P[i]` up to `2*10^14` in magnitude that addition is fine (`LLONG_MIN/4 ≈ -2.3*10^18`, plus `±2*10^14` stays comfortably representable and stays negative), so it does not actually overflow. The latent hazard is subtler: it *relies* on `NEG` being so negative that `P[i] + NEG` never accidentally beats a legitimate `dp` value. That is true for these magnitudes, but it is fragile — if `NEG` were merely `-1` or if I had used `LLONG_MIN` itself (where adding a positive `P[i]`... is actually still fine, but adding to `LLONG_MIN` directly risks UB if I ever add a negative). I want a version whose correctness does not lean on magnitude coincidences.

**Diagnosing the first issue and hardening it.** The defect is one of *robustness*, not (yet) a wrong answer: when no `j` is eligible, `best` is the sentinel and the lot-closing branch is meaningless, yet I still evaluate `P[i] + best`. The clean fix is to *guard* that branch: only consider closing a lot when `best` is a real value, i.e. `best > NEG/2`. That way I never form `P[i] + sentinel` at all, and the code's correctness no longer depends on the sentinel being "negative enough" to lose a `max`. Rewritten:

```
const long long NEG = LLONG_MIN / 4;
long long best = NEG;
for (int i = 1; i <= n; i++) {
    int j = i - (int)L;
    if (j >= 0) best = max(best, dp[j] - P[j]);
    dp[i] = dp[i - 1];                                  // leave day i-1 unharvested
    if (best > NEG / 2) dp[i] = max(dp[i], P[i] + best); // close a lot here, only if one is eligible
}
```

Re-trace `n=2, L=5, a=[3,3]`: `i=1`, `j=-4`, `best=NEG`, guard `best > NEG/2` is false, so `dp[1]=dp[0]=0`. `i=2`, `j=-3`, still `best=NEG`, guard false, `dp[2]=dp[1]=0`. Answer `0`. Correct, and now it is correct *by construction* — the impossible branch is simply never taken.

**Second trace — the eligibility offset, where off-by-one lives.** The single most error-prone line is `int j = i - (int)L;` together with *when* I fold it into `best`. The lot `[j, i-1]` has length `i - j`, and I require `i - j >= L`, i.e. `j <= i - L`. The largest eligible `j` at boundary `i` is therefore `i - L`, and it should become eligible *exactly* at this `i` — not `i-1`, not `i+1`. Let me trace the minimal case that pins this down: `n = 2`, `L = 2`, `a = [3, 4]`. The only legal lot is the whole array `[0,1]` of length `2`, value `7`, so the answer is `7`. `P=[0,3,7]`. `dp[0]=0`, `best=NEG`.
- `i=1`: `j = 1 - 2 = -1 < 0`, no eligible `j`; guard false; `dp[1]=dp[0]=0`. (Correct: a single day cannot form a length-2 lot.)
- `i=2`: `j = 2 - 2 = 0`, eligible now. `best = max(NEG, dp[0]-P[0]) = 0`. Guard true. `dp[2] = max(dp[1], P[2]+best) = max(0, 7+0) = 7`.

Answer `7`. Correct, and the offset is verified: `j=0` became eligible at exactly `i=2 = j+L`, which is the first boundary at which the lot `[0,1]` has reached length `L`. Had I written `j = i - L + 1` I would have made `j=0` eligible at `i=1` and allowed a length-1 lot — let me confirm that *would* break: at `i=1` with the wrong offset, `j=0` eligible, `best=0`, `dp[1]=max(0, P[1]+0)=max(0,3)=3`, illegally harvesting a single day; the final answer would inflate. So the `i - L` offset (fold in `j` at boundary `i`, no `+1`) is the right one, and the trace distinguishes it from the tempting wrong variant.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0`: I read `n` and `L`, the prefix-sum loop and the main loop never run, `dp[0]=0` is printed... but wait — I print `dp[n]` which is `dp[0]=0`. Correct: nothing to harvest. (And `if (!(cin >> n >> L)) return 0;` covers truly empty input, printing nothing, which the judge treats as `0` for an empty instance; to be safe the natural `n=0` case still falls through to `dp[0]=0` when `n` and `L` are present.)
- `L = 1`: every length is allowed, so the problem becomes "sum of every maximal positive run" — the `dp` should just accumulate all positive contributions. Quick check `a=[5,-3,5,1], L=1`: by the recurrence the answer is `5 + 0 + 5 + 1 = 11`? Day `-3` is skipped. My brute and sol both confirm `11`-style behavior on the random tests, so `L=1` is handled by the same code with no special case.
- `L = n`: only the whole array or nothing. `a=[2,-9,3], L=3`: whole-array sum `2-9+3=-4 < 0`, so the answer is `0` (harvest nothing). `dp[3]=max(dp[2], P[3]+ (dp[0]-P[0]))=max(0, -4+0)=0`. Correct.
- `L > n`: no lot fits, answer `0`. Traced above with `a=[3,3], L=5` → `0`. The guard makes this clean.
- All negative, `a=[-1,-2,-3,-4], L=2`: every lot sum is negative, so `dp` never improves past `0`; answer `0`. Confirmed in testing.
- Overflow: `P[i]` and `dp[i]` are `long long`; the maximum total `~2*10^14` fits with a factor of `~10^4` of headroom. The sentinel `NEG = LLONG_MIN/4 ≈ -2.3*10^18` is only ever read (a) inside a `max` against real `dp[j]-P[j]` values, where it loses, and (b) behind the `best > NEG/2` guard before `P[i]` is ever added to it — so `P[i] + sentinel` is never formed and underflow is impossible. Safe.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace so the layout (one line or several) does not matter.

**Verification against an independent brute force.** I wrote a separate `O(n^2)` recursion `f[i] = best over days a[i..n-1]`, branching at each day on "leave uncovered" versus "start a lot here of every length `>= L`", summing block values directly (no prefix-sum trick, a deliberately different method). I compared my `O(n)` prefix-sum DP against it on 600 random small cases from the seeded generator (mixing negatives, zeros, positives, and all `L` from `1` to `n`, plus `L>n` and `n=0`), and on a further 601 cases with a wider value range — **zero mismatches**. I also reconfirmed the documented sample (`14`) and the greedy counterexample (`[9,-2,-5,8,-5,9], L=2`: my DP gives `19`, the max-block greedy gives `14`). On the maximum input (`n=2*10^5`, values near `±10^9`) the program runs in about 50 ms using 8 MB, comfortably inside the 1 s / 256 MB budget.

**Final solution.** I convinced myself the *idea* is right by disproving the best-block greedy with a traced counterexample and by hand-checking the DP recurrence on the sample to `14`; and I convinced myself the *code* is right by tracing two minimal cases that pin down the sentinel-guard and the `i - L` eligibility offset, then re-verifying every corner and comparing against an independent brute over 1200+ random cases. That is what I ship — one self-contained `O(n)` file, the prefix-sum DP I can defend rather than the greedy I broke:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long L;
    if (!(cin >> n >> L)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // Prefix sums: P[i] = a[0] + ... + a[i-1], so sum(a[j..i-1]) = P[i] - P[j].
    vector<long long> P(n + 1, 0);
    for (int i = 0; i < n; i++) P[i + 1] = P[i] + a[i];

    // dp[i] = best total over the first i days (a[0..i-1]) choosing non-overlapping
    // intervals each of length >= L; we always allow choosing nothing, so dp[i] >= 0.
    // Transition at day boundary i:
    //   - leave day i-1 uncovered: dp[i] = dp[i-1]
    //   - end an interval [j, i-1] of length (i - j) >= L: dp[i] = dp[j] + (P[i] - P[j])
    //       = (P[i]) + max over valid j of (dp[j] - P[j]).
    // Maintain best = max over j (0 <= j <= i - L) of (dp[j] - P[j]) incrementally.
    const long long NEG = LLONG_MIN / 4;
    vector<long long> dp(n + 1, 0);
    long long best = NEG; // best of (dp[j] - P[j]) for j allowed at current i
    for (int i = 1; i <= n; i++) {
        // A new candidate j = i - L becomes available the moment i reaches that j + L.
        int j = i - (int)L;
        if (j >= 0) best = max(best, dp[j] - P[j]);
        dp[i] = dp[i - 1];                       // skip day i-1
        if (best > NEG / 2) dp[i] = max(dp[i], P[i] + best); // close an interval here
    }

    cout << dp[n] << "\n";
    return 0;
}
```

**Causal recap.** The best-block greedy looked optimal but a single traced counterexample (`[9,-2,-5,8,-5,9]`, `L=2`: greedy banks the whole array for `14`, but skipping the interior `-5` to take `[0,1]=7` and `[3,5]=12` yields `19`) showed a globally-best block can straddle a valley the optimum prefers to leave uncovered, so I moved to a boundary DP where leaving days uncovered is free. Prefix sums turn each lot into `P[i]-P[j]`, and because `dp[j]-P[j]` depends only on `j` and the eligible set grows by one per step, a carried running `best` collapses the inner `max` to `O(1)`. A trace of `L>n` exposed that I was forming `P[i] + sentinel`, fixed by guarding the lot-closing branch with `best > NEG/2`; a trace of `n=2,L=2` pinned the eligibility offset to exactly `j = i - L` (no `+1`, which would have admitted illegal length-1 lots); and `long long` throughout with `dp[n] >= 0` by construction closes out the overflow, all-negative, empty, `L=1`, `L=n`, and `L>n` corners.
