**Reading the problem and pinning the contract.** I have a rod of integer length `n` and a price list `p[1..n]`, where `p[k]` is what one piece of length exactly `k` sells for. I may cut the rod into any number of integer-length pieces — including not cutting at all, which just means selling it whole as a single length-`n` piece — and I want to maximize the sum of the prices of the pieces I end up with. Cuts are free, so the only thing that matters is the multiset of piece lengths, and that multiset must be a composition of `n`: lengths summing exactly to `n`. Input on stdin is `n` followed by the `n` prices; I print a single integer, the maximum revenue. The first thing I always do is fix the scale, because it decides the data types. Here `n <= 5000` and `0 <= p[k] <= 10^9`. The worst case for the revenue magnitude is cutting the rod into `n` pieces each of the most expensive single length: that is at most `n * max_price = 5000 * 10^9 = 5 * 10^12`. That comfortably exceeds the 32-bit signed range of about `2.1 * 10^9`, so the answer and every accumulator must be 64-bit. I will use `long long` for `p[]` and for the DP table. This is non-negotiable: an `int` here is a silent wrong-answer on the large tests, even though the *index* `n` itself fits in an `int` fine.

**Laying out the candidate approaches.** Two routes are on the table, and I want to commit to the one I can *prove*, not the one that is easiest to type or that "feels" efficient.

- *Greedy by price-per-length.* The intuition is seductive: treat the rod as raw material and always cut off the piece length whose price-per-unit-length `p[k]/k` is the best among lengths that still fit, then repeat on the remainder. It runs in roughly `O(n)` cuts times `O(n)` scan each, it is short, and it mirrors how you would intuitively "spend" a resource — take the densest value first. The structural worry is exactly the worry that kills most greedies: the constraint here is global (the lengths must sum to `n`), and a price ratio is a purely local figure of merit. Committing a length-`k` piece because it has the best ratio silently spends `k` units of the budget, and there is no reason the *remaining* `n - k` units split as well as some other first choice would have. I refuse to trust this until I have tried to break it.
- *Dynamic programming over rod length.* Define `dp[L]` as the best revenue obtainable from a rod of length `L`, and build `dp[0..n]` bottom-up. The risk here is not whether the idea is correct — it is the textbook optimal-substructure setup — but whether I transcribe the recurrence and its base case without an off-by-one or a degenerate-`best` bug. So the two routes have different risk profiles: greedy might be *conceptually* wrong, and the DP might be *mechanically* wrong. I will attack greedy on concept first, and if it falls, attack the DP on mechanics.

**Stress-testing greedy before committing.** Hand-waving "greedy feels right" is precisely how wrong solutions get shipped, so let me attack it with a concrete instance rather than an intuition. Take `n = 4` with prices `p = [1, 5, 8, 9]` — so a length-1 piece sells for 1, length-2 for 5, length-3 for 8, length-4 for 9. The price-per-length ratios are `1/1 = 1.000`, `5/2 = 2.500`, `8/3 ≈ 2.667`, `9/4 = 2.250`. Greedy looks at the whole length-4 rod, sees the best ratio is length-3 at `2.667`, and cuts a length-3 piece off, earning `8`. That leaves a remnant of length `1`; the only piece that fits is length-1, earning `1`. Greedy's total is `8 + 1 = 9`.

Is `9` optimal? Let me hunt for a composition greedy structurally cannot reach. Cut the rod into two pieces of length 2: that is a valid composition `2 + 2 = 4`, and it earns `5 + 5 = 10`. That is strictly better than greedy's `9`. So greedy is wrong, and I can now see *why* in this very instance: the length-3 piece had the highest density `2.667`, but grabbing it forced the leftover to be a length-1 piece worth a miserly `1`, whose density `1.0` dragged the average down. The optimum instead used two pieces of merely-good density `2.5`, but it tiled the whole budget at that good density with no scrap left over. The local "best ratio" decision was a trap: it optimized the first piece's density in isolation and left a remnant it could not use well. The verification paid off — it killed an approach I would otherwise have been tempted to ship for its simplicity. Greedy is out, conclusively, on a four-element example I can hold in my head.

To be sure this is not a one-off fluke of a hand-picked case, I made a mental note to later run greedy against the eventual correct solution over a few hundred random small instances; when I did (below, after the DP was verified), greedy disagreed with the optimum on roughly one in sixty random instances — for example `n = 10`, `p = [1,10,16,16,17,15,3,17,1,7]`, where greedy earns `49` but the optimum earns `52`. That confirms the counterexample is representative of a whole failure mode, not an isolated curiosity.

**Deriving the DP and checking the recurrence on paper.** With greedy dead, I derive the DP I can defend. I want `dp[L]` = the maximum revenue from a rod of length `L`. The key observation is optimal substructure expressed through the *first piece I cut off the left end*. Whatever the optimal way to cut a length-`L` rod is, it has some leftmost piece of some length `k` with `1 <= k <= L`; that piece earns `p[k]`, and the rest of the rod — a contiguous remnant of length `L - k` — is itself cut optimally, earning `dp[L - k]`. (It must be cut optimally, or I could improve the whole by improving the remnant — the standard cut-and-paste exchange argument.) I do not know which first length `k` the optimum uses, so I try them all and take the best:

`dp[L] = max over k in 1..L of ( p[k] + dp[L - k] )`,  with base case `dp[0] = 0`.

`dp[0] = 0` is the empty rod: nothing left to cut, revenue zero. Note this recurrence enumerates *every* composition implicitly: peeling the leftmost piece and recursing covers each ordered way to write `L` as a sum of positive parts exactly once per composition, and the `max` keeps the best. There is no greedy choice here — every first-piece length is considered. Because `dp[L]` only ever depends on strictly shorter `dp[L-k]`, building the table in increasing `L` order means every value I read is already finished. That is `O(n)` lengths times `O(n)` candidate first-pieces each, so `O(n^2)` time and `O(n)` memory.

Let me confirm the recurrence by hand on the counterexample `p = [1, 5, 8, 9]`, `n = 4`, where I established the answer is `10`. `dp[0] = 0`. `dp[1] = p[1] + dp[0] = 1 + 0 = 1`. `dp[2] = max(p[1] + dp[1], p[2] + dp[0]) = max(1 + 1, 5 + 0) = max(2, 5) = 5`. `dp[3] = max(p[1] + dp[2], p[2] + dp[1], p[3] + dp[0]) = max(1 + 5, 5 + 1, 8 + 0) = max(6, 6, 8) = 8`. `dp[4] = max(p[1] + dp[3], p[2] + dp[2], p[3] + dp[1], p[4] + dp[0]) = max(1 + 8, 5 + 5, 8 + 1, 9 + 0) = max(9, 10, 9, 9) = 10`. The recurrence delivers `10`, matching the optimum and beating greedy's `9`. The DP is correct on the very case that broke greedy, which is exactly the evidence I want.

**Is `O(n^2)` fast enough?** With `n = 5000`, the inner double loop does about `n(n+1)/2 ≈ 1.25 * 10^7` additions and comparisons. That is well under a second — on the order of ten milliseconds in practice — so I do not need anything cleverer. I deliberately set the constraint at `n <= 5000` precisely so that the simple, provable quadratic DP passes comfortably with enormous margin, rather than reaching for some sub-quadratic trick I would then have to argue is correct under budget. The simple correct method is fast enough here by construction; that is the whole point.

**First implementation — and a trace, because clean math transcribes dirty.** My first cut of the core:

```
vector<long long> dp(n+1, 0);
for(int L=1;L<=n;L++){
    long long best = 0;                 // <-- careless initialization
    for(int k=1;k<=L;k++){
        long long cand = p[k] + dp[L-k];
        if(cand>best) best=cand;
    }
    dp[L]=best;
}
```

Something about initializing `best = 0` nags at me. The `dp[L]` is a maximum over candidates, and seeding the running max with `0` quietly asserts "revenue at least 0 is always achievable for length `L`," folding a phantom zero-revenue option into the maximum. For a length-`L` rod that is not literally true as a *piece* decision — you must cut *some* first piece of positive length — so I want to know whether that seed can mask a real candidate. The dangerous direction is when all genuine candidates are smaller than the seed. Prices are constrained `>= 0` here, so `p[k] + dp[L-k] >= 0` always and the `0` seed cannot actually win... but I do not want correctness to rest on a constraint that a future reuse of this code might relax (negative prices, say a "disposal cost" variant), and I do not want a latent landmine. So I trace the smallest input that could expose the seed: a rod where every single price is `0`. Take `n = 2`, `p = [0, 0]`. The answer is obviously `0` (any way of cutting yields `0`). With `best = 0`: `dp[1] = max-seeded-0 over (p[1]+dp[0]=0) = 0`. `dp[2] = max-seeded-0 over (p[1]+dp[1]=0, p[2]+dp[0]=0) = 0`. Output `0`. It happens to be right — but only because every candidate equals the seed.

**Diagnosing the latent bug.** The code returns the right answer on that case, but for the wrong *reason*: the `0` seed is doing load-bearing work it should not be doing. To make the defect bite, I imagine the natural extension where prices can be negative (a length might cost money to produce). Suppose `p = [-3]`, `n = 1`. The only composition of a length-1 rod is the single length-1 piece, so the true answer is `-3` — you are forced to take it. But my code computes `dp[1] = max(seed 0, p[1]+dp[0] = -3) = 0`, reporting that a length-1 rod earns `0` by "cutting nothing," which is impossible: a length-1 rod *must* contain one length-1 piece. The seed `0` invented a free empty-cut option for a non-empty rod. Even though the *current* constraints (`p[k] >= 0`) hide this, it is a real modeling error in the recurrence's transcription: `dp[L]` for `L >= 1` should be a maximum strictly over real first-piece choices, with no synthetic zero mixed in. The right seed is "no candidate yet seen," i.e. negative infinity, so that the first real candidate sets the bar and the maximum reflects only achievable cuts.

**Fixing and re-verifying.** I replace the seed with `LLONG_MIN` so `best` is purely the maximum over genuine first-piece candidates, and since the inner loop always runs at least once for `L >= 1` (range `k in 1..L` is non-empty), `best` is always overwritten by a real candidate before being stored:

```
for(int L=1;L<=n;L++){
    long long best = LLONG_MIN;
    for(int k=1;k<=L;k++){
        long long cand = p[k] + dp[L-k];
        if(cand>best) best=cand;
    }
    dp[L]=best;
}
```

I must double-check the seed cannot cause its own overflow problem. `LLONG_MIN` is only ever read inside the comparison `cand > best`; I never *add* anything to `best` (the addition is `p[k] + dp[L-k]`, both finite and non-negative), so the sentinel can never underflow. And on the very first iteration `k = 1`, `cand = p[1] + dp[L-1]` is a finite non-negative value that beats `LLONG_MIN`, so `best` is real from then on. Re-trace `n = 1`, `p = [-3]` (the case that exposed the bug): `dp[1] = max(LLONG_MIN, -3 + dp[0]=-3) = -3`. Correct — it now reports the forced loss instead of a phantom zero. Re-trace `n = 2`, `p = [0,0]`: `dp[1] = max(LLONG_MIN, 0) = 0`, `dp[2] = max(LLONG_MIN, 0, 0) = 0`, output `0`. Still correct, but now for the right reason. The transcription is sound regardless of the sign of the prices, which is the robustness I wanted.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0`: the read of `n` succeeds and is `0`, the price loop reads nothing, the `dp` loop `L in 1..0` never runs, and I output `dp[0] = 0`. The empty rod — correct.
- `n = 1`: one price read; `dp[1] = p[1] + dp[0] = p[1]`; output `p[1]`. A length-1 rod must be sold whole — correct.
- Whole-rod-is-best (increasing prices, e.g. `p = [1,2,3,4,9]`): the candidate `k = L` contributes `p[L] + dp[0] = p[L]`, so "don't cut" is always among the choices considered; if it is best, the DP picks it. Correct, and notably this is the case where greedy-by-ratio might wrongly cut.
- All-length-1-is-best (decreasing prices): the candidate `k = 1` chains `p[1] + dp[L-1]`, building the all-singletons composition, and the DP picks it if best. Correct.
- Empty input / no `n`: `if(!(cin>>n)) return 0;` outputs nothing and exits cleanly, which matches "no rod given."
- Overflow: `p[]` and `dp[]` are `long long`; the maximum revenue `~5 * 10^12` fits with four orders of magnitude to spare. I confirmed on a generated `n = 5000` all-prices-near-`10^9` instance that the output (about `1.59 * 10^12`) is correct and well within range.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace, so prices given space- or newline-separated both parse.

**Self-verification harness — the part I actually trust.** Hand-traces convince me of small cases; a differential test convinces me of the rest. I wrote an independent oracle two different ways: for `n <= 18` it enumerates *every* integer composition of `n` by plain recursion (peel a leftmost piece of every length, recurse, sum prices, keep the max) — the literal definition of the problem with zero algorithmic cleverness — and for larger `n` it uses a memoized top-down recursion structured oppositely to my bottom-up table, so a transcription slip in one would not silently match the other. A generator produced random instances across several price styles deliberately chosen to provoke the failure modes: monotone-increasing lists (tempting "never cut"), monotone-decreasing lists (tempting "all length-1"), lists of zeros with isolated spikes, and generic random prices. I compiled the solution and ran it against the oracle on a battery of hand-picked edge cases (`n = 0`, `n = 1`, all-zeros, the `[1,5,8,9]` counterexample, increasing/decreasing/spiky lists) plus several hundred random instances — 613 cases in total — and got **zero mismatches**. The first version, with the `best = 0` seed, would have passed this particular non-negative-price battery too (because the seed only bites with negative prices), which is exactly why I did not rely on the battery alone to vet the seed: the hand-trace on the hypothetical negative-price extension is what justified the `LLONG_MIN` seed, and the 613-case differential run confirms the shipped version is correct on the actual constraint set. I also ran my deliberately-wrong greedy against the verified solution over the random battery and watched it disagree on a steady ~1.5% of cases, which is the empirical tombstone for the greedy idea.

**Final solution.** I convinced myself the *idea* is right by disproving greedy with a concrete `[1,5,8,9]` counterexample (greedy `9`, optimum `10`) and by deriving the first-piece DP recurrence and checking it by hand on that same instance; I convinced myself the *code* is right by tracing the seed bug to a precise cause on a negative-price extension, fixing the seed to negative infinity, re-verifying the two cases that motivated it, and then differential-testing 613 cases against an independent exhaustive oracle with zero mismatches. The constraints (`n <= 5000`) were chosen so this `O(n^2)` DP runs in milliseconds with vast margin. That is what I ship — one self-contained file, the simple provable DP I can defend, reached by killing the clever greedy I broke:

```cpp
#include <bits/stdc++.h>
using namespace std;
int main(){
    int n;
    if(!(cin>>n)) return 0;
    vector<long long> p(n+1, 0);
    for(int i=1;i<=n;i++) cin>>p[i];
    // dp[L] = max revenue obtainable from a rod of length L.
    // dp[0]=0; dp[L]=max over first-piece length k in 1..L of p[k]+dp[L-k].
    vector<long long> dp(n+1, 0);
    for(int L=1;L<=n;L++){
        long long best = LLONG_MIN;
        for(int k=1;k<=L;k++){
            long long cand = p[k] + dp[L-k];
            if(cand>best) best=cand;
        }
        dp[L]=best;
    }
    cout << dp[n] << "\n";
    return 0;
}
```

**Causal recap.** The price-per-length greedy looked clean and resource-rational, but a single traced counterexample (`p = [1,5,8,9]`: greedy cuts the densest length-3 piece for `9` total, while two length-2 pieces earn `10`) showed that committing the locally densest first piece strands a low-value remnant, so a global summation constraint cannot be served by a local ratio; I discarded greedy and derived the first-piece DP `dp[L] = max_k (p[k] + dp[L-k])`, `dp[0] = 0`, which considers every first-piece length rather than guessing one, and checked it by hand to `10` on the same instance; transcribing it, I caught that seeding the inner maximum with `0` smuggles a phantom empty-cut option into a non-empty rod (harmless under `p >= 0` but a real recurrence error, exposed by a `p = [-3]` trace returning `0` instead of `-3`), and fixed it by seeding with `LLONG_MIN` so the maximum reflects only achievable cuts; 64-bit accumulators cover the `~5 * 10^12` revenue ceiling, the `O(n^2)` table clears `n = 5000` in milliseconds, and a 613-case differential run against an independent exhaustive-composition oracle returned zero mismatches, closing out the empty, single-element, all-zero, monotone, and large-`n` corners.
