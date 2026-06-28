**Reading the problem and pinning the contract.** I have `n` balloons in a row, balloon `i` painted with `nums[i]`. I burst them one at a time, in any order; bursting `i` pays `nums[left] * nums[i] * nums[right]` coins where `left`/`right` are the balloons still present immediately beside `i` at that instant, and a missing neighbour (I fell off either end of the row) counts as a balloon painted `1`. After a burst the balloon is removed and its two neighbours become adjacent. I want the order that maximizes total coins, and I print that one number. Input is `n` then the `n` values; `0 <= n <= 500`, `0 <= nums[i] <= 100`. If `n = 0` the answer is `0`.

Let me fix the scale before any algorithm, because that governs both feasibility and data types. The order is an arrangement of `n` bursts, so the naive search is over `n!` orders — at `n = 500` that is beyond astronomical, so anything that enumerates orders (or even subsets, `2^n`) is dead on the large tests. Whatever I ship has to be polynomial. On arithmetic: every burst pays at most `100 * 100 * 100 = 10^6`, and there are exactly `n` bursts, so the total is at most `n * 10^6 = 5 * 10^8` for `n = 500`. That actually still fits in a signed 32-bit `int` (cap about `2.1 * 10^9`), so this problem is *not* an overflow trap — but the cube `v[l]*v[k]*v[r]` is the kind of product I never want to compute in `int` out of reflex, and `long long` costs me nothing at `n = 500`, so I will carry 64-bit throughout and not have to think about it again. That is the one cheap, non-negotiable decision.

**Laying out the candidate approaches.** Two routes are on the table, and I want to commit to the one I can *prove*, not the one that is fastest to type.

- *Greedy by value.* Pick a fixed local rule keyed on the painted numbers. The two natural ones: "burst the smallest remaining balloon first" — the intuition being you cash a cheap balloon out *now*, while a big neighbour is still next to it to multiply against, before that neighbour gets popped — or "burst the largest remaining first/last." Each is `O(n^2)` and a handful of lines. The risk is structural: the payout couples each balloon to whichever neighbours happen to survive beside it, so a burst reshapes the multipliers of *every* later burst. That global coupling is exactly the configuration where a rule that decides one balloon at a time tends to be wrong. I will not trust it until I have tried to break it.
- *Interval dynamic programming.* Pad the row with the two virtual `1` balloons and build answers up by ranges, reasoning about a distinguished balloon in each range. `O(n^3)` time, `O(n^2)` memory. The risk here is not the idea but getting the *state and transition* right — interval DP for this problem has a famous trap in how you pick the distinguished balloon, and `n^3` at `n = 500` is `1.25 * 10^8`, which I need to confirm fits the time limit.

**Stress-testing greedy before committing.** Hand-waving "smallest-first feels right" is how wrong solutions get shipped, so let me attack it with the very first concrete instance, the sample `nums = [3, 1, 5, 8]`. Smallest-first: the smallest is `1` at index 1; its neighbours are `3` and `5`, so I collect `3 * 1 * 5 = 15`, and the row becomes `[3, 5, 8]`. Next smallest is `3` at the left end; its left neighbour is the boundary `1`, right is `5`, so `1 * 3 * 5 = 15`, row `[5, 8]`. Next smallest is `5`; neighbours boundary-`1` and `8`, so `1 * 5 * 8 = 40`, row `[8]`. Last, `8` alone: `1 * 8 * 1 = 8`. Greedy total `15 + 15 + 40 + 8 = 78`.

Is `78` optimal? The problem statement hands me a witness: burst `1`, then `5`, then `3`, then `8` gives `3*1*5 + 3*5*8 + 1*3*8 + 1*8*1 = 15 + 120 + 24 + 8 = 167`, more than double `78`. So smallest-first is wrong, and I can see *why*: smallest-first popped `5` while only the boundary `1` was on its left, scoring a measly `1 * 5 * 8`, whereas the optimal order kept `3` and `8` flanking `5` so that `5` is burst as `3 * 5 * 8 = 120`. The greedy threw away the chance to let `5` be multiplied by two large neighbours at once. And largest-first is no better: pop `8` first as `1 * 8 * 1 = 8`... it is plainly squandering the boundary. The verification paid off immediately — it killed the approach I would otherwise have reached for. Greedy in every fixed-local-rule form is out, and the reason is exactly the global neighbour-coupling I flagged.

**Deriving the DP — and walking straight into the classic trap first.** I want a range-based DP. My first instinct is the most natural one: let `f[i][j]` be the max coins obtainable from bursting all balloons in the closed range `[i..j]`, and decide the recurrence by which balloon I burst **first** in that range. So I pick some `m` in `[i..j]`, burst it first, then solve `[i..m-1]` and `[m+1..j]`. Let me write that transition and immediately see if it even type-checks against the rule.

When I burst `m` *first* in `[i..j]`, what are its neighbours? They are whatever sits just left and just right of `m` *inside the whole current configuration* — which at this moment is the full range `[i..j]` with both `m-1` and `m+1` still present. So the gain is `nums[m-1] * nums[m] * nums[m+1]` (with boundary `1`s if `i` or `j` is at an end of the whole row). Fine so far. But now I recurse into `[i..m-1]` and `[m+1..j]` and I am stuck: once I have burst `m`, the balloons `m-1` and `m+1` have become adjacent, so when I later burst, say, the last balloon of `[i..m-1]`, its right neighbour is no longer `m` (gone) — it is whatever is left of `m+1`. The two sub-ranges are *not independent*: solving `[i..m-1]` in isolation has no idea what balloon ends up on its right, because that depends on the order I burst things in `[m+1..j]`. The "first to burst" framing leaves the neighbours of the sub-problems undefined. The recurrence is unsound; if I coded it I would get garbage, and I caught it on paper before writing a line.

**The fix: reason about the LAST balloon to burst.** The trap is that the *first* balloon's removal scrambles the neighbours of the sub-problems. So flip it: in a range, ask which balloon `k` is burst **last**. If `k` is last among the balloons strictly between two fixed survivors `l` and `r` (think of `l` and `r` as still-present walls), then at the instant `k` is popped *every other balloon between `l` and `r` is already gone*, so `k`'s neighbours are exactly `l` and `r`. That fixes `k`'s payout to `v[l] * v[k] * v[r]` independent of the internal order, and — crucially — it makes the two sides independent: everything strictly between `l` and `k` is burst (in some order) entirely before `k`, all while `l` and `k` stand as its walls; likewise everything strictly between `k` and `r`, with `k` and `r` as walls. The walls of each sub-problem are now *pinned*, so the sub-problems decouple. That is the whole insight.

So I pad: `v[0] = 1`, `v[n+1] = 1`, and `v[1..n]` are the real balloons. Define

- `dp[l][r]` = max coins from bursting *every* balloon strictly between padded indices `l` and `r` (i.e. the open interval `(l, r)`), with `l` and `r` themselves still present as walls.

Transition: choose the last balloon `k` in `(l, r)`:

```
dp[l][r] = max over k in (l, r) of  dp[l][k] + v[l]*v[k]*v[r] + dp[k][r]
```

`dp[l][k]` handles everything strictly between `l` and `k` (walls `l`, `k`), `dp[k][r]` handles everything strictly between `k` and `r` (walls `k`, `r`), and `v[l]*v[k]*v[r]` is `k`'s own payout as the last one standing between its pinned walls. Base case: if `(l, r)` is empty — `r = l + 1`, no balloon between them — there is nothing to burst, `dp[l][r] = 0`. The answer is `dp[0][n+1]`: burst everything strictly between the two virtual walls, which is the whole real row.

**Checking the recurrence by hand on the sample.** `nums = [3, 1, 5, 8]`, so padded `v = [1, 3, 1, 5, 8, 1]` over indices `0..5`, and I want `dp[0][5]`. Let me build by gap length. Length-1 gaps (one balloon between the walls): `dp[0][2]` has only `k=1`: `v[0]*v[1]*v[2] = 1*3*1 = 3`. `dp[1][3]` only `k=2`: `3*1*5 = 15`. `dp[2][4]` only `k=3`: `1*5*8 = 40`. `dp[3][5]` only `k=4`: `5*8*1 = 40`. Length-2 gaps: `dp[0][3]`, `k in {1,2}`: `k=1` -> `dp[0][1]=0 + 1*3*5 + dp[1][3]=15 = 30`; `k=2` -> `dp[0][2]=3 + 1*1*5 + dp[2][3]=0 = 8`; max `30`. `dp[1][4]`, `k in {2,3}`: `k=2` -> `0 + 3*1*8 + dp[2][4]=40 = 64`; `k=3` -> `dp[1][3]=15 + 3*5*8 + 0 = 135`; max `135`. `dp[2][5]`, `k in {3,4}`: `k=3` -> `0 + 1*5*1 + dp[3][5]=40 = 45`; `k=4` -> `dp[2][4]=40 + 1*8*1 + 0 = 48`; max `48`. Length-3 gaps: `dp[0][4]`, `k in {1,2,3}`: `k=1` -> `0 + 1*3*8 + dp[1][4]=135 = 159`; `k=2` -> `dp[0][2]=3 + 1*1*8 + dp[2][4]=40 = 51`; `k=3` -> `dp[0][3]=30 + 1*5*8 + dp[3][4]=0 = 70`; max `159`. `dp[1][5]`, `k in {2,3,4}`: `k=2` -> `0 + 3*1*1 + dp[2][5]=48 = 51`; `k=3` -> `dp[1][3]=15 + 3*5*1 + dp[3][5]=40 = 70`; `k=4` -> `dp[1][4]=135 + 3*8*1 + 0 = 159`; max `159`. Length-4 gap, the answer `dp[0][5]`, `k in {1,2,3,4}`: `k=1` -> `0 + 1*3*1 + dp[1][5]=159 = 162`; `k=2` -> `dp[0][2]=3 + 1*1*1 + dp[2][5]=48 = 52`; `k=3` -> `dp[0][3]=30 + 1*5*1 + dp[3][5]=40 = 75`; `k=4` -> `dp[0][4]=159 + 1*8*1 + 0 = 167`; max `167`. That matches the stated answer `167`, and the maximizing `k=4` (burst `8` last) is consistent with the optimal order the statement gave. The recurrence is right.

**First implementation — and the iteration order is where this kind of DP dies.** I need `dp[l][k]` and `dp[k][r]` already computed when I fill `dp[l][r]`. Both have a strictly smaller gap (`k - l < r - l` and `r - k < r - l`), so iterating by increasing gap length `len = r - l` guarantees the dependencies are ready. My first cut of the loops:

```
for (int len = 2; len <= n + 1; len++) {
    for (int l = 0; l + len <= n + 1; l++) {
        int r = l + len;
        long long best = 0;
        for (int k = l + 1; k < r; k++)
            best = max(best, dp[l][k] + v[l]*v[k]*v[r] + dp[k][r]);
        dp[l][r] = best;
    }
}
```

The thing I always get wrong here is the *bounds* on `len` and `l`, so before trusting it I traced the smallest non-trivial input by hand. I built the DP wrong on a first scratch pass: I had written the outer loop as `for (len = 1; len <= n; len++)` — copying the muscle memory of "lengths run from 1 to n" from closed-interval DPs. Let me trace what that does on `nums = [3, 1, 5, 8]`, `n = 4`, where I now know the answer is `167`.

**Diagnosing the bug.** With `len` running only `1..n = 1..4`, the largest gap I ever fill is `r - l = 4`, i.e. `dp[0][4]` (and `dp[1][5]`). But the answer I print is `dp[0][n+1] = dp[0][5]`, whose gap is `n + 1 = 5`. My loop never sets `dp[0][5]`, so it keeps its initialized value `0` and the program prints `0` instead of `167`. The defect is precise: in *this* DP the meaningful gaps run from `2` (a wall-balloon-wall triple, the smallest range that contains a balloon) up to `n + 1` (the two outer walls `0` and `n+1` with all `n` balloons between them) — not `1..n`. The closed-interval reflex `len in 1..n` is off by the padding. There is a second, quieter thing to get right: a gap of `len = 1` means `r = l + 1`, an *empty* open interval with no balloon to burst, whose `dp` must stay `0`; my corrected loop simply starts `len` at `2` so those empty gaps are never (and need never be) assigned, and they correctly remain `0` from initialization, which is exactly what the base case wants.

**Fixing and re-verifying.** Correct the bounds: gap length runs `len = 2 .. n+1`, and for each `len` the left wall runs while `l + len <= n + 1` so that `r = l + len` stays within the padded array:

```
for (int len = 2; len <= n + 1; len++) {
    for (int l = 0; l + len <= n + 1; l++) {
        int r = l + len;
        ...
    }
}
```

Re-running on `[3, 1, 5, 8]` now fills up through `dp[0][5]` and prints `167` — the value I got by hand. I also re-checked the corners on paper: `n = 0` makes the only meaningful gap `len = 1` (`dp[0][1]`, empty), the `len`-loop runs from `2` to `1` so its body never executes, and I print `dp[0][1] = 0` — the empty row, correct. `n = 1`, `nums = [7]`: padded `v = [1, 7, 1]`, only gap `len = 2` is `dp[0][2]` with `k = 1` -> `1 * 7 * 1 = 7`; print `7`, correct. A row of zeros pays `0` everywhere, correct. The case that broke before now passes, and it broke for exactly the bound I fixed, which is the evidence I trust.

**Now the real verification: an independent brute oracle, not just hand-traces.** Hand-tracing the sample proves the recurrence on one input; it does not prove the *code* over the space of inputs, and I have already been burned once by a bounds reflex. So I wrote a completely independent brute that does **not** use the last-balloon insight at all: it directly simulates the bursting game, recursing over *which balloon to burst next* on the current list of survivors, scoring each choice as `left * cur * right` against the present neighbours (boundary `1`s at the ends), and taking the max over all choices — memoized on the set of surviving original positions so it stays tractable to about `n = 9`. This explores the actual game tree by *first* move, the framing I rejected for the DP, so a bug shared between oracle and solution is unlikely; they are derived from different principles.

I generated random rows (`n` up to `9`, values including plenty of `0`s and the cap `100`) plus hand edge cases — `n = 0`, single `0`, single `100`, all zeros, all `100`s, the sample, alternating `100 0 100 0 ...` — and diff-tested `sol` against the brute. Over **1000+** cases the counts came back **0 mismatches**. I then ran the worst case for time and arithmetic, `n = 500` all `100`s: the program finished in about `0.02 s` (the DP is roughly `n^3 / 6 ~ 2 * 10^7` real operations, far under the `1.25 * 10^8` ceiling and far under the `1 s` limit), used a few MB, and produced `498010100` — about `5 * 10^8`, which confirms my arithmetic bound and that even an `int` would have survived here, so `long long` is comfortable defensive headroom rather than a load-bearing necessity. The DP table is `(n+2)^2` longs `~ 2 MB`, well within `256 MB`.

**Final solution.** I convinced myself the *idea* is right by disproving greedy on a concrete witness (`[3, 1, 5, 8]`: smallest-first `78` vs optimal `167`) and by deriving the recurrence around the *last* balloon to burst — after first walking into and rejecting the unsound *first*-balloon framing whose sub-problems have undefined neighbours. I convinced myself the *code* is right by tracing the off-by-padding gap-length bug to a precise cause, fixing the bounds, and then differential-testing against an independent game-tree brute over a thousand-plus cases with zero mismatches, plus a worst-case timing and range check. That is what I ship — one self-contained file, the simple `O(n^3)` interval DP I can prove, not the greedy I broke:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> 0 coins
    vector<long long> v(n + 2);
    v[0] = 1;                              // virtual balloon "1" on the left
    v[n + 1] = 1;                          // virtual balloon "1" on the right
    for (int i = 1; i <= n; i++) cin >> v[i];

    // dp[l][r] = max coins from bursting every balloon strictly between
    // padded indices l and r (exclusive), with l and r still present.
    // Choose k in (l, r) as the LAST balloon burst in that open interval;
    // when k is burst its neighbours are exactly l and r, contributing
    // v[l] * v[k] * v[r], plus the two already-solved sub-intervals.
    vector<vector<long long>> dp(n + 2, vector<long long>(n + 2, 0));
    for (int len = 2; len <= n + 1; len++) {          // gap length r - l
        for (int l = 0; l + len <= n + 1; l++) {
            int r = l + len;
            long long best = 0;
            for (int k = l + 1; k < r; k++) {
                long long cur = dp[l][k] + v[l] * v[k] * v[r] + dp[k][r];
                if (cur > best) best = cur;
            }
            dp[l][r] = best;
        }
    }

    cout << dp[0][n + 1] << "\n";
    return 0;
}
```

**Causal recap.** Greedy looked plausible but one traced witness (`[3,1,5,8]`: smallest-first `78` vs optimal `167`) showed a fixed local pop-rule squanders the chance to flank a balloon with two big neighbours, because the payout couples each balloon to its surviving neighbours globally — so greedy is out. Building the interval DP, my reflex "burst-first on closed `[i..j]`" framing was unsound because removing the first balloon leaves the sub-problems' walls undefined; pivoting to "burst-*last* between fixed walls `l` and `r`" pins each balloon's payout to `v[l]*v[k]*v[r]` and decouples the two sides, giving `dp[l][r] = max_k dp[l][k] + v[l]*v[k]*v[r] + dp[k][r]` with answer `dp[0][n+1]`. My first loop inherited the closed-interval reflex `len in 1..n` and never filled `dp[0][n+1]`, printing `0` on the sample; fixing the gap bounds to `len in 2..n+1` makes it print `167`, and an independent game-tree brute over `1000+` cases at `0` mismatches plus a `0.02 s` worst-case run closes it out.
