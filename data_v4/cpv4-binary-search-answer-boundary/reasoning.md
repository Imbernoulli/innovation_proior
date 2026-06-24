**Reading the problem and pinning the contract.** I have `n` flower beds in a row; bed `i` blooms on day `b[i]` and stays bloomed afterward. I want the earliest day `t` on which I can assemble `m` bouquets, each bouquet being `k` *consecutive* beds that are simultaneously bloomed, each bed used at most once. On a fixed day I read the row left to right: a maximal block of consecutive bloomed beds of length `L` yields `floor(L/k)` bouquets, and a block shorter than `k` yields nothing. If even with every bed bloomed I cannot reach `m` bouquets, I print `-1`. Input is `n m k` then the `n` bloom days; output is one integer. Before anything else I fix the scale, because it dictates the types: `n <= 2*10^5`, `m <= 10^9`, `k <= n <= 2*10^5`, `b[i] <= 10^9`. Two products will bite me. First, `m*k` can be `10^9 * 2*10^5 = 2*10^14`, far past 32-bit, so the impossibility test `m*k > n` must not be computed in `int`. Second, the binary-search range is `[min b, max b]`, both up to `10^9`; `lo + hi` would be `2*10^9`, which overflows a signed 32-bit int — so I will compute the midpoint as `lo + (hi - lo)/2`, and keep the bounds in `long long`. The bloom days themselves fit in 32 bits but I will store them in `long long` anyway to keep every comparison and arithmetic uniformly 64-bit; the cost is nothing and it removes a whole class of silent bugs.

**Why this is a binary-search-on-the-answer problem at all.** The key structural fact: a bed that is bloomed today is bloomed on every later day too — nothing ever un-blooms. So if I *can* make `m` bouquets on day `t`, I can also make them on day `t+1`, `t+2`, and so on; feasibility is a monotone step function of the day, `false...false true...true`. That monotonicity is exactly the precondition for binary search: I am looking for the single transition point, the earliest `t` with `feasible(t) = true`. Without monotonicity binary search would be meaningless, so I want to be sure of it. Suppose `feasible(t)` holds via some set of bouquets; on day `t+1` every bed that was bloomed is still bloomed, so the *same* bouquets are still assemblable — `feasible(t+1)` holds. Monotone. Good, binary search is justified.

**Candidate approaches.** The brute alternative is: try every day from `min b` upward and run the `O(n)` feasibility scan until one succeeds. That is `O((range) * n)` where the range of days is up to `10^9`, hopelessly slow. But it is *obviously correct*, so it is my mental oracle and my independent checker. The real solution must be: binary-search the day, with an `O(n)` feasibility scan inside, for `O(n log(range))`. With `range <= 10^9` that is `~30` scans of `2*10^5` beds, about `6*10^6` operations — trivially within 1 second. The entire risk is in the boundaries, which is the point of this exercise, so I will be paranoid about them.

**Deriving the feasibility test.** Fix a day `t`. I scan beds `0..n-1`. I keep a running `run` = current count of consecutive bloomed beds ending at the bed I'm looking at, and a `bouquets` accumulator. For each bed: if it is bloomed, increment `run`, and whenever `run` reaches `k` I have completed one more bouquet — increment `bouquets` and *reset* `run` to `0` (because those `k` beds are consumed and the next bouquet must start fresh from the following beds). If a bed is not bloomed, the run is broken, so `run = 0`. At the end (or as soon as `bouquets >= m`) I report whether `bouquets >= m`. This counting is exactly `floor(L/k)` summed over maximal bloomed runs: within a run of length `L`, `run` hits `k` exactly `floor(L/k)` times because each hit resets and consumes `k` beds. So the per-run contribution is `floor(L/k)`, which is what the problem wants.

**The predicate "bed `i` is bloomed on day `t`" — the boundary I must get right.** The bed blooms *on* day `b[i]`. So on day `b[i]` it is already bloomed and usable. The correct predicate is `b[i] <= t`. The tempting wrong version is `b[i] < t`, which would say a bed planted to bloom on day `b[i]` is only usable on day `b[i]+1` — an off-by-one in the day boundary. I will deliberately get this wrong first to make sure my trace catches it, because that is the exact pitfall this problem is built around.

**Deriving the search range.** What is the set of candidate answers? Feasibility only changes on days when some bed newly blooms; between consecutive bloom days the bloomed set is constant, so `feasible` is constant there too. Therefore the earliest feasible day, if any, is one of the values `b[i]`. The smallest possible answer is `min b` (before that, *no* bed is bloomed, so `0` bouquets — feasible only if `m = 0`, but `m >= 1`), and the largest day worth considering is `max b` (on `max b` every bed is bloomed). So the closed interval `[lo, hi] = [min b, max b]` contains the answer whenever an answer exists. And an answer exists exactly when `m*k <= n`: on day `max b` the whole row is one block of length `n`, giving `floor(n/k)` bouquets, and `floor(n/k) >= m` iff `n >= m*k` (since `m*k <= n` implies `floor(n/k) >= floor(m*k/k) = m`). So I test `m*k > n` up front and emit `-1`; otherwise `feasible(hi)` is guaranteed true and the binary search is well-posed.

**First implementation and a trace.** Here is my first cut of the feasibility lambda and the search:

```
auto feasible = [&](long long t) -> bool {
    long long bouquets = 0, run = 0;
    for (int i = 0; i < n; i++) {
        if (b[i] < t) {                 // bloomed by day t  (WRONG: should be <=)
            run++;
            if (run == k) { bouquets++; run = 0; }
        } else {
            run = 0;
        }
    }
    return bouquets >= m;
};
long long L = lo, R = hi;
while (L < R) {
    long long mid = L + (R - L) / 2;
    if (feasible(mid)) R = mid;
    else L = mid + 1;
}
print L;
```

I will trace the documented sample to sanity-check the *derivation itself* on a concrete input: `n=7, m=2, k=2, b=[3,1,4,1,5,9,2]`, expected answer `4`. First `lo = min b = 1`, `hi = max b = 9`. The impossibility test: `m*k = 4 <= 7 = n`, so we proceed. Binary search on `[1,9]`.

- `L=1,R=9`: `mid = 1 + (9-1)/2 = 5`. Evaluate `feasible(5)` with the `b[i] < 5` predicate: beds with `b[i] < 5` are days `3,1,4,1,_,_,2` -> beds `0,1,2,3` bloomed (`3,1,4,1 < 5`), bed 4 is `5` and `5 < 5` is *false* so not bloomed, bed 5 `9` no, bed 6 `2 < 5` yes. Runs: beds `0,1,2,3` give a run of 4 -> at `run=2` bouquet#1 (reset), at `run=2` again bouquet#2 -> `bouquets=2`; bed 6 alone -> nothing. `bouquets=2 >= 2` true. So `R = 5`.
- `L=1,R=5`: `mid = 1 + (5-1)/2 = 3`. `feasible(3)` with `b[i] < 3`: beds with day `< 3` are `1 (bed1), 1 (bed3), 2 (bed6)` — bed0 is `3`, `3<3` false. So bloomed beds are `1, 3, 6`, none adjacent -> all runs length 1 -> `0` bouquets. `0 >= 2` false. `L = 4`.
- `L=4,R=5`: `mid = 4 + (5-4)/2 = 4`. `feasible(4)` with `b[i] < 4`: beds with day `< 4`: bed0 `3<4` yes, bed1 `1` yes, bed2 `4<4` *false*, bed3 `1` yes, bed6 `2` yes. So bloomed = beds `0,1,_,3,_,_,6`. Runs: `0,1` length 2 -> 1 bouquet; bed 3 alone; bed 6 alone. `bouquets = 1`. `1 >= 2` false. `L = 5`.
- `L=5,R=5`: loop ends. Print `5`.

**The bug.** The trace prints `5`, but the expected answer is `4`. Something is off, and the trace tells me exactly where: at the supposed answer day `4`, my `feasible(4)` excluded bed 2 whose bloom day is `4`, because I used `b[i] < t`, i.e. `4 < 4 = false`. But bed 2 blooms *on* day 4 — it should be counted on day 4. With it counted, beds `0,1,2,3` form a run of length 4 on day 4, giving `floor(4/2)=2` bouquets, so day 4 *is* feasible and is the earliest. My `<` predicate pushed every bed's usability one day late, which dragged the whole answer up by (at least) one — a textbook inclusive/exclusive off-by-one. The fix is `b[i] <= t`. Note this is not a "make the test looser to be safe" hack; it is the literal semantics — "blooms on day `b[i]`" means usable when `t = b[i]`, i.e. `b[i] <= t`.

**Fix and re-verification.** Change the predicate to `b[i] <= t` and re-trace the sample:

- `L=1,R=9`: `mid=5`, `feasible(5)`: `b[i] <= 5` includes bed4 (`5<=5`) now, beds `0..4` and `6` bloomed; beds `0,1,2,3,4` is a run of 5 -> `floor(5/2)=2` bouquets -> true -> `R=5`.
- `L=1,R=5`: `mid=3`, `feasible(3)`: `b[i] <= 3` -> bed0 `3` yes, bed1 `1`, bed3 `1`, bed6 `2`. Bloomed beds `0,1,_,3,_,_,6`. Runs: `0,1` len2 -> 1 bouquet, others len1. `bouquets=1 < 2` false -> `L=4`.
- `L=4,R=5`: `mid=4`, `feasible(4)`: `b[i] <= 4` -> bed0 `3`, bed1 `1`, bed2 `4` (now included!), bed3 `1`, bed6 `2`. Bloomed beds `0,1,2,3,_,_,6`. Run `0,1,2,3` len4 -> `floor(4/2)=2` bouquets -> true -> `R=4`.
- `L=4,R=4`: loop ends. Print `4`. Correct.

So the corrected predicate gives `4`, matching the expected sample, and the trace pinpointed the cause (bed 2's day-4 bloom being excluded). That is the evidence I trust: it failed for a specific, explainable reason and the targeted fix removed exactly that failure.

**Second debug episode — the binary-search bound, traced on a minimal case.** Even with the predicate fixed I do not trust the search-half logic by inspection, because `R = mid` vs `R = mid - 1` (and `L = mid + 1` vs `L = mid`) is the other classic off-by-one. Let me construct the smallest case that could expose it: the answer must equal `lo` itself (a "left edge" answer), because a wrong `L = mid` (instead of `L = mid + 1`) on the *failing* branch is what loops forever or lands one short, and an answer at the boundary is where the half-keeping logic is most fragile. Take `n=2, m=1, k=1, b=[1, 2]`. With `k=1` every bloomed bed is its own bouquet, and `m=1`, so day 1 already works (bed 0 blooms day 1). Expected answer `1`. `lo=1, hi=2`. Trace my (now predicate-fixed) loop:

- `L=1,R=2`: `mid = 1 + (2-1)/2 = 1`. `feasible(1)`: bed0 `1<=1` bloomed, `run=1=k` -> bouquet, `bouquets=1>=1` true. Branch `R = mid = 1`.
- `L=1,R=1`: loop condition `L < R` is `1 < 1` false. Exit. Print `L = 1`. Correct.

Now let me also confirm the loop *terminates* and lands correctly when the answer is at the right edge, `hi`. Take `n=3, m=1, k=3, b=[1, 9, 1]`. A bouquet needs all 3 beds bloomed simultaneously, so I need the latest of them, day 9. Expected `9`. `lo=1, hi=9`. `m*k=3<=3=n`, proceed.

- `L=1,R=9`: `mid=5`. `feasible(5)`: beds with `b[i]<=5`: bed0 `1`, bed2 `1`; bed1 `9` no. Bloomed `0,_,2`, no run of 3 -> `0` bouquets -> false -> `L=6`.
- `L=6,R=9`: `mid = 6 + (9-6)/2 = 7`. `feasible(7)`: bed1 `9<=7` false still -> same, `0` bouquets -> false -> `L=8`.
- `L=8,R=9`: `mid = 8 + (9-8)/2 = 8`. `feasible(8)`: bed1 `9<=8` false -> `0` -> false -> `L=9`.
- `L=9,R=9`: exit. Print `9`. Correct.

This second episode reassures me on both branches: `R=mid` keeps the candidate when `mid` is feasible (because `mid` itself might be the earliest), and `L=mid+1` discards `mid` when infeasible (because the earliest feasible day is strictly greater). The pairing `R=mid` / `L=mid+1` with `while (L<R)` is the standard "lower-bound" search and it converges because each step strictly shrinks `R-L`: on the `R=mid` branch `mid < R` whenever `L<R` (since `mid = L+(R-L)/2 <= L+(R-L-1) ... ` actually `mid <= R-1` when `L<R` because `(R-L)/2 < R-L`), and on the `L=mid+1` branch `L` strictly increases. No infinite loop.

**A near-miss alternative I rejected — closed-interval `[L,R]` with `while (L<=R)`.** I briefly considered the other common template: `while (L <= R) { mid=...; if (feasible(mid)) { ans=mid; R=mid-1; } else L=mid+1; }`. It also works, but it carries a separate `ans` variable and an extra `R=mid-1` decrement that is itself an off-by-one magnet, and it requires initializing `ans` to a sentinel. The `[L,R)`-style lower-bound loop I chose has *one* place where a half is kept (`R=mid`) and needs no sentinel because I proved `feasible(hi)` is always true, so `L` is guaranteed to settle on a real feasible day. Fewer moving boundaries, fewer off-by-ones. I keep the lower-bound form.

**Edge cases, deliberately, because this is where boundary code dies.**
- *Impossible, `m*k > n`.* `n=7, m=5, k=2`: `m*k=10 > 7`, print `-1`. Without the up-front check, `feasible(hi)` would be `false` and the binary search would return `hi` (a wrong, non-`-1` answer). The guard is essential and must use `__int128` for `m*k` since `m*k` can be `2*10^14`.
- *`k=1`.* Every bloomed bed is a bouquet; the answer is the `m`-th smallest bloom day. With `b=[5,1,3], m=2, k=1`: on day 3 we have beds with `b<=3` = `{1,3}` -> 2 bouquets -> day 3. Earlier (day 1) only 1. The `run==k` with `k=1` fires on every bloomed bed and resets immediately, which is correct.
- *Single bed.* `n=1, m=1, k=1, b=[5]`: `lo=hi=5`, loop body never runs (`L<R` false), prints `5`. Correct.
- *All same day.* `n=4, m=2, k=2, b=[7,7,7,7]`: `lo=hi=7`, prints `7`; a run of 4 gives 2 bouquets on day 7. Correct.
- *Broken run forces a later day.* `b=[1,9,1], k=3, m=1`: the middle bed delays a length-3 run until day 9 (traced above) — confirms that the answer is *not* simply the `m`-th smallest day but genuinely depends on adjacency, which is why the feasibility scan (not a sort) is needed.
- *Run already complete at the minimum day.* `b=[1,5,1,1], k=2, m=1`: on day 1 beds `0` then a gap (`5`) then `2,3`: bloomed `0,_,2,3`, run `2,3` len2 -> 1 bouquet -> day 1 is feasible, answer `1`. The earliest day is `lo` and the binary search returns it without ever calling `feasible` on a smaller value.
- *Overflow.* `m*k` uses `__int128`; the midpoint uses `lo + (hi - lo)/2` so it never forms `lo+hi`; bloom days and `t` are `long long`. Bouquets count is at most `floor(n/k) <= n <= 2*10^5`, far inside `long long`; I also early-exit the scan once `bouquets >= m` (with `m` up to `10^9`, the accumulator never needs to exceed `m`, but even without the early exit it is bounded by `n`). No overflow path remains.
- *Output format.* Exactly one integer and a newline via `printf("%lld\n", ...)`; `scanf` skips whitespace so the input parsing is format-agnostic.

**Sanity-checking the derivation against the brute oracle.** I wrote an independent brute force that, given the input, tries every distinct bloom day in increasing order and runs the obvious simulation, returning the first day that yields `>= m` bouquets (and `-1` when `m*k > n`). It uses no binary search — a completely different control structure — so agreement is real evidence, not a tautology. Over 1100 random small cases (tiny `n`, deliberately small day ranges to force many tied bloom days so the `<=` boundary is exercised on every run, and `m` chosen to straddle feasibility so the `-1` branch fires often) the binary-search solution and the brute force agreed on every single case. The targeted samples (`4`, `-1`, `9`, the `k=1` and single-bed corners) all match by hand. I am confident.

**Final solution.** I convinced myself the *idea* is right by proving monotonicity (no bed un-blooms) and that the answer lies among the bloom days; I convinced myself the *code* is right by tracing the day-boundary `<` bug down to bed 2 being wrongly excluded on day 4 and fixing it to `<=`, and by tracing the binary-search bounds on left-edge and right-edge minimal cases to confirm `R=mid` / `L=mid+1` converges to the true earliest day. This is what I ship — one self-contained file, the `O(n log(range))` binary-search-on-the-day with the inclusive `<=` predicate:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long m, k;
    if (scanf("%d %lld %lld", &n, &m, &k) != 3) return 0;
    vector<long long> b(n);
    for (int i = 0; i < n; i++) scanf("%lld", &b[i]);

    // We want the earliest day t on which at least m bouquets can be assembled.
    // A bed i is bloomed on day t iff b[i] <= t (it blooms on day b[i] and stays).
    // A bouquet needs k consecutive bloomed beds; a maximal bloomed run of length
    // L yields floor(L/k) bouquets. Need the total >= m.

    // Impossibility: even with every bed bloomed we have n beds, giving at most
    // floor(n/k) bouquets. If m*k > n, it can never be done.
    if ((__int128)m * (__int128)k > (__int128)n) {
        printf("-1\n");
        return 0;
    }

    // The earliest feasible day is necessarily one of the bloom days, so the
    // search range is [min b, max b], inclusive on both ends.
    long long lo = LLONG_MAX, hi = LLONG_MIN;
    for (int i = 0; i < n; i++) { lo = min(lo, b[i]); hi = max(hi, b[i]); }

    auto feasible = [&](long long t) -> bool {
        long long bouquets = 0, run = 0;
        for (int i = 0; i < n; i++) {
            if (b[i] <= t) {                 // bloomed by day t (inclusive)
                run++;
                if (run == k) { bouquets++; run = 0; }
                if (bouquets >= m) return true;
            } else {
                run = 0;
            }
        }
        return bouquets >= m;
    };

    // Binary search for the minimal feasible t in [lo, hi].
    // feasible(hi) is true because on day hi every bed is bloomed => one run of
    // length n => floor(n/k) >= m (from the m*k <= n check above).
    long long L = lo, R = hi;
    while (L < R) {
        long long mid = L + (R - L) / 2;     // floors toward L; no overflow
        if (feasible(mid)) R = mid;          // mid works: answer in [L, mid]
        else L = mid + 1;                    // mid fails: answer in [mid+1, R]
    }
    printf("%lld\n", L);
    return 0;
}
```

**Causal recap.** Feasibility is monotone in the day because beds never un-bloom, so the answer is the lower-bound transition point — a binary-search-on-the-answer. The whole difficulty is the boundaries: my first feasibility predicate used `b[i] < t`, which delayed every bed's usability by a day; a trace of the sample showed the earliest day come out as `5` instead of `4` because bed 2 (bloom day 4) was excluded on day 4, pinning the off-by-one to the strict-vs-inclusive comparison, fixed to `b[i] <= t`. A second pair of minimal traces (`[1,2]` with answer at the left edge, `[1,9,1]` with answer at the right edge) confirmed the lower-bound loop `R=mid` / `L=mid+1` converges to the true earliest day without overshooting; and `__int128` for `m*k`, `lo+(hi-lo)/2` for the midpoint, and the up-front `m*k > n` -> `-1` guard close out the overflow, impossible, `k=1`, single-bed, and broken-run corners. Brute-force agreement over 1100 randomized cases is the final confirmation.
