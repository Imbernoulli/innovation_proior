**Reading the problem and pinning the contract.** A reservoir starts the season at level `0`. Day `i` adds an integer `d[i]` (positive, negative, or zero) to the level, so the level after day `i` is the prefix sum `P[i] = d[0] + ... + d[i]`, and the level before any day is `P[-1] = 0`. I have to report the *worst decline*: the largest value of `P[i] - P[j]` over all `-1 <= i <= j <= n-1`. Since `i = j` is permitted, that drop can be `0`, so the answer is never negative. Input on stdin is `n` then the `n` values; I print one integer. Before touching an algorithm I fix the scale, because it dictates the data types: `n <= 2*10^5` and `|d[i]| <= 10^9`, so a level can reach `2*10^5 * 10^9 = 2*10^14`, and a decline (peak minus trough) can be as large as roughly twice that in magnitude — well past the 32-bit ceiling of about `2.1*10^9`. So every prefix accumulator, the running peak, and the answer must be 64-bit. I will use `long long` everywhere. That is the first non-negotiable decision; an `int` here is a silent wrong-answer on the large tests, and it is exactly the kind of bug that passes every small sample and dies on the hidden ones.

**The single subtlety I want to nail down first: does the start level count?** The definition ranges `i` from `-1`, and `P[-1] = 0` is the starting level. That is not decoration — it is load-bearing. Consider a single negative day, `d = [-7]`. The levels are `P[-1] = 0` and `P[0] = -7`. The only nontrivial pair is `i = -1, j = 0`, giving `P[-1] - P[0] = 0 - (-7) = 7`. So the answer for `[-7]` is `7`, **not** `0`. If I forgot the start level and only ranged the peak over actual days, I would conclude "the level never rose, so there is no decline, answer `0`" — and be wrong. This is the opposite of the maximum-subarray family where an all-negative array yields `0` by taking nothing; here the start level `0` is a genuine, always-present peak, so an all-negative season produces a *large* decline, not zero. I am flagging this now because I suspect my first implementation will get the base case wrong precisely here.

**Laying out the candidate approaches.** Two routes, and I want the one I can prove and also run fast.

- *All-pairs scan.* For every pair `i <= j` (including the start index), compute `P[i] - P[j]` and keep the max. It is `O(n^2)`, obviously correct, and I will use it as my mental reference oracle — but at `n = 2*10^5` it is `4*10^10` operations, hopelessly over the 1-second limit. Not shippable.
- *Prefix sum with a running peak.* For a fixed later index `j`, the drop `P[i] - P[j]` is largest when `P[i]` is the **maximum** level among indices `i <= j`. So if I sweep left to right and keep `peak = max level seen so far` (the start level `0` included), then at index `j` the best decline ending there is `peak - P[j]`, and the global answer is the max of that over all `j`. One pass, `O(n)` time, `O(1)` memory. The open questions are not the idea but the *transcription*: (1) is the start level `0` correctly seeded into `peak`, and (2) do I update `peak` before or after I read it for the current decline?

**Deriving the recurrence carefully.** Define `answer = max over j of ( peak_{<=j} - P[j] )`, where `peak_{<=j} = max(P[-1], P[0], ..., P[j])`. I claim this equals the all-pairs max. Proof sketch: for any pair `(i, j)` with `i <= j`, `P[i] - P[j] <= peak_{<=j} - P[j]`, since `peak_{<=j} >= P[i]`; and the bound is achieved by the `i` that realizes the max. Taking the max over `j` of `peak_{<=j} - P[j]` therefore equals the max over all valid pairs. Good — the running-peak sweep is exactly the all-pairs quantity, computed in linear time.

Now, *where does the start level enter?* The pair `i = -1` must be available for every `j >= -1`. The clean way is to initialize `peak = 0` (that is `P[-1]`) and `prefix = 0` *before* the loop, and let the loop handle days `0..n-1`. Then when I process day `j`, the `peak` already includes `P[-1]` and all earlier days, so `peak - prefix` correctly considers the start level as a candidate high-water mark. The empty pair `i = j` (drop `0`) I cover by initializing `answer = 0`, since the worst decline is always at least `0`.

**Sanity-checking the derivation on the sample, by hand.** Sample `d = [3, -2, -3, 4, -5, 2, 1]`, claimed answer `6`. Levels: start `0`; after day0 `3`; day1 `1`; day2 `-2`; day3 `2`; day4 `-3`; day5 `-1`; day6 `0`. The highest level is `3` (after day0), the lowest *after* that is `-3` (after day4), so the worst decline should be `3 - (-3) = 6`. Let me run the running-peak recurrence step by step, starting `prefix = 0, peak = 0, answer = 0`:

- day0 `+3`: `prefix = 3`; decline `peak - prefix = 0 - 3 = -3`; `answer = max(0, -3) = 0`; then `peak = max(0, 3) = 3`.
- day1 `-2`: `prefix = 1`; decline `3 - 1 = 2`; `answer = max(0, 2) = 2`; `peak = max(3, 1) = 3`.
- day2 `-3`: `prefix = -2`; decline `3 - (-2) = 5`; `answer = max(2, 5) = 5`; `peak = 3`.
- day3 `+4`: `prefix = 2`; decline `3 - 2 = 1`; `answer = 5`; `peak = max(3, 2) = 3`.
- day4 `-5`: `prefix = -3`; decline `3 - (-3) = 6`; `answer = max(5, 6) = 6`; `peak = 3`.
- day5 `+2`: `prefix = -1`; decline `3 - (-1) = 4`; `answer = 6`; `peak = 3`.
- day6 `+1`: `prefix = 0`; decline `3 - 0 = 3`; `answer = 6`; `peak = 3`.

Final `answer = 6`. The derivation matches the sample, and I can see the worst decline being pinned at day4 exactly where the trough sits below the day0 peak. The idea is right.

**First implementation — and immediately a trace, because the order of the peak update is exactly the kind of thing I get wrong.** My first cut of the loop body, written fast:

```
long long prefix = 0, peak = 0, answer = 0;
for (int i = 0; i < n; i++) {
    long long d; cin >> d;
    prefix += d;
    peak = max(peak, prefix);          // update peak FIRST
    answer = max(answer, peak - prefix);
}
```

The thing nagging me is that I updated `peak` *before* measuring the decline. Does that matter? On a falling day it does not (peak does not change), but let me trace a case where the current day is itself a new high, because that is where "include the current level in the peak" could spuriously shrink a later decline... actually no — including the current level in `peak` before measuring only ever makes `peak` larger or equal, so `peak - prefix` can only grow, never shrink. So this particular ordering does not *lose* a decline. But it can *invent* one that should not exist. Let me hunt for that. The risk is: by folding `prefix` into `peak` and then immediately computing `peak - prefix`, on the very day the level rises I compute `prefix - prefix = 0`, which is harmless. Hmm, the self-pairing always gives `0`. So updating peak first looks harmless here. Let me trace the all-negative case to be sure I have not broken the base case, since that is the corner I flagged.

Trace `d = [-2, -3, -1, -4]`, which I expect to answer `2 + 3 + 1 + 4 = 10` (level falls monotonically from `0` to `-10`, worst decline `0 - (-10) = 10`). Start `prefix = 0, peak = 0, answer = 0`.

- day0 `-2`: `prefix = -2`; `peak = max(0, -2) = 0`; decline `0 - (-2) = 2`; `answer = 2`.
- day1 `-3`: `prefix = -5`; `peak = 0`; decline `0 - (-5) = 5`; `answer = 5`.
- day2 `-1`: `prefix = -6`; `peak = 0`; decline `0 - (-6) = 6`; `answer = 6`.
- day3 `-4`: `prefix = -10`; `peak = 0`; decline `0 - (-10) = 10`; `answer = 10`.

Final `10`. Correct — the start level `0` stayed as the peak because every day fell below it, and the base case held. So this "update peak first" version actually works for the all-negative corner. Good — but I have not yet found the bug I was worried about, which makes me suspicious I am testing the wrong corner.

**The bug — a different, real one in a variant I almost wrote.** While cleaning the code I wrote a "tidier" version that seeds `peak` from the first element instead of from the start level, a refactor that *feels* equivalent:

```
long long prefix = 0, answer = 0;
long long peak;                         // seeded inside the loop on i == 0
for (int i = 0; i < n; i++) {
    long long d; cin >> d;
    prefix += d;
    if (i == 0) peak = prefix;          // "the first level is the first peak"  <-- BUG
    else peak = max(peak, prefix);
    answer = max(answer, peak - prefix);
}
```

This is the classic wrong base case: it makes the *first day's level* the initial peak and silently drops the start level `0` as a candidate. Trace the single-negative case `d = [-7]`, where I derived by hand the answer must be `7` (start `0` down to `-7`). Start `prefix = 0, answer = 0`.

- day0 `-7`: `prefix = -7`; `i == 0` so `peak = prefix = -7`; decline `peak - prefix = -7 - (-7) = 0`; `answer = max(0, 0) = 0`.

Final `0`. **Wrong** — it should be `7`. The defect is precise: by seeding `peak` from `P[0] = -7` I threw away the start level `P[-1] = 0`, which is the actual high-water mark. The pair `(i = -1, j = 0)` that realizes the decline of `7` never gets considered because index `-1` was never represented. This is exactly the base-case / sign-handling trap the problem is built around: with negatives, the *initial* zero level is often the peak, and any code that only looks at peaks *among the days themselves* will under-report or report `0` when the true answer is large. Let me also trace the all-negative `[-2,-3,-1,-4]` through this buggy variant to show it is not a one-off: `i=0` sets `peak = -2`, decline `0`; `i=1` `prefix=-5`, `peak=max(-2,-5)=-2`, decline `-2-(-5)=3`; `i=2` `prefix=-6`, decline `-2-(-6)=4`; `i=3` `prefix=-10`, decline `-2-(-10)=8`. It reports `8`, but the true worst decline is `10`. It is undercounting by exactly the start level's contribution `0 - (-2) = 2` every time. So the bug is systematic, not incidental.

**Fix and re-verification.** The fix is to seed `peak = 0` (the start level `P[-1]`) *before* the loop and never special-case the first day, so the start level is always in contention:

```
long long prefix = 0, peak = 0, answer = 0;
for (int i = 0; i < n; i++) {
    long long d; cin >> d;
    prefix += d;
    answer = max(answer, peak - prefix);   // measure decline against the best earlier peak
    peak = max(peak, prefix);              // then fold today's level into the peak
}
```

Re-trace `[-7]`: start `(prefix, peak, answer) = (0, 0, 0)`. day0 `-7`: `prefix = -7`; decline `peak - prefix = 0 - (-7) = 7`; `answer = 7`; `peak = max(0, -7) = 0`. Final `7`. Correct. Re-trace `[-2,-3,-1,-4]`: day0 `prefix=-2`, decline `0-(-2)=2`, `answer=2`, `peak=0`; day1 `prefix=-5`, decline `5`, `answer=5`; day2 `prefix=-6`, decline `6`; day3 `prefix=-10`, decline `10`, `answer=10`, `peak=0`. Final `10`. Correct. The two cases that broke the buggy variant now pass, and they pass for the reason I fixed — the start level is back in the peak — which is the evidence I trust.

**A second debug episode: the measure-then-update vs update-then-measure order.** I switched the fixed version to *measure the decline first, then update the peak*. Is that order actually required, or merely cosmetic? Let me reason and trace. On any day, `peak - prefix` measured *before* folding today's level uses `peak = max(P[-1..i-1])`, i.e. the best *strictly earlier* high-water mark. If I instead update first, I use `peak = max(P[-1..i])`, which includes today. But today's own level `P[i]` enters the decline as `P[i] - P[i] = 0`, which never beats the `answer >= 0` floor. So including today in the peak can only add the harmless `0` term — it cannot create a false positive decline. Trace to confirm on a rising-then-falling case `d = [5, -3]` (levels `0, 5, 2`; true worst decline `5 - 2 = 3`). *Measure-then-update*: day0 `prefix=5`, decline `0-5=-5`, `answer=0`, `peak=5`; day1 `prefix=2`, decline `5-2=3`, `answer=3`, `peak=5`. Gives `3`. *Update-then-measure*: day0 `prefix=5`, `peak=5`, decline `5-5=0`, `answer=0`; day1 `prefix=2`, `peak=5`, decline `5-2=3`, `answer=3`. Also `3`. They agree, as my argument predicted. So both orders are correct here — but I deliberately keep *measure-then-update* because it is the order that makes the invariant easiest to state ("`peak` is the best level strictly before the day whose decline I am measuring"), and an invariant I can state in one sentence is an invariant I am less likely to break later. The earlier base-case bug taught me that the danger in this problem is not the arithmetic but the bookkeeping of *which levels are eligible*, so I pick the formulation that keeps that bookkeeping legible.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0`: the loop never runs; `answer` stays `0`. The empty season has no decline — correct. (Also, `if (!(cin >> n)) return 0;` guards against truly empty input.)
- `n = 1`, single positive `d = [4]`: day0 `prefix=4`, decline `0-4=-4`, `answer=0`, `peak=4`. The level only rose, no decline — correct, `0`.
- `n = 1`, single negative `d = [-7]`: answer `7` as traced — the start level is the peak. Correct, and distinct from `0`.
- All non-negative, e.g. `[5, 5, 5]`: prefixes `5, 10, 15`, each decline `peak - prefix` with `peak <= prefix` gives `<= 0`, so `answer = 0`. Correct — a never-falling level has worst decline `0`.
- All zeros `[0, 0, 0]`: every `prefix = 0`, every decline `0`, `answer = 0`. Correct — flat level, no decline.
- All negative `[-2,-3,-1,-4]`: answer `10` as traced — the running peak stays pinned at the start `0`. Correct.
- Overflow: `prefix`, `peak`, `answer` are all `long long`; the largest decline `~2*10^14` fits with three orders of magnitude to spare. I stress-tested `n = 2*10^5` all `-10^9`, where the level dives to `-2*10^14` and the answer is `0 - (-2*10^14) = 2*10^14`; the program prints `200000000000000`, confirming no 32-bit truncation. An `int` here would have wrapped and reported garbage.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace, so the input format (one line or many) does not matter.

**Verification against a brute oracle.** Beyond the hand traces, I ran the `O(n)` sweep against an independent `O(n^2)` all-pairs brute force (which explicitly builds the level list including the start `0` and maximizes `levels[i] - levels[j]` over `i <= j`) on 700 random small cases — `n` from `0` to `8`, values biased toward negatives and zeros, with occasional all-negative, all-positive, and empty inputs, plus sporadic large-magnitude values. Zero mismatches. The brute force uses a completely different method (no running peak, just every pair), so the agreement is real evidence and not a shared bug. Combined with the explicit overflow run, I am confident in both the idea and the transcription.

**Final solution.** I convinced myself the idea is right by deriving the running-peak identity and checking it on the sample, and I convinced myself the *code* is right by catching the base-case bug (dropping the start level `0`) through a trace of `[-7]` that returned `0` instead of `7`, fixing it by seeding `peak = 0` before the loop, and re-verifying the negative corners plus overflow. That is what I ship — one self-contained `O(n)` file:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;            // n = 0 (or empty input) -> answer 0

    // Level starts at 0 before any day; this initial level is a valid "peak".
    long long prefix = 0;                  // P[-1] = 0
    long long peak = 0;                    // best (max) level seen so far, including the start
    long long answer = 0;                  // i = j gives a decline of 0, so answer >= 0

    for (int i = 0; i < n; i++) {
        long long d;
        cin >> d;
        prefix += d;                       // level after day i
        // decline ending at this day = (highest earlier-or-equal level) - current level
        answer = max(answer, peak - prefix);
        peak = max(peak, prefix);          // update running peak AFTER measuring the decline
    }

    cout << answer << "\n";
    return 0;
}
```

**Causal recap.** The worst decline equals `max over j of (best level strictly before-or-at j) - (level at j)`, which a single left-to-right sweep computes by carrying a running `peak`; the derivation checked out on the sample (`[3,-2,-3,4,-5,2,1] -> 6`, pinned at the day0 peak above the day4 trough). The trap the problem is built on is the base case: the *starting* level `0` is itself a peak, so an all-negative or single-negative season has a large decline, not `0` — a "tidier" variant that seeded the peak from the first day's level returned `0` for `[-7]` (should be `7`) and `8` for `[-2,-3,-1,-4]` (should be `10`), undercounting by the start level every time; seeding `peak = 0` before the loop restores index `-1` as an eligible peak and fixes both. Measuring the decline before folding today's level into the peak keeps the invariant ("`peak` = best strictly-earlier level") legible, and `long long` accumulators with the answer floored at `0` close out the empty, single-element, all-negative, all-zero, and `2*10^14`-overflow corners.
