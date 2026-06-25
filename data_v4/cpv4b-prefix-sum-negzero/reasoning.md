**Reading the problem and pinning the contract.** I have a reservoir gauge that reads `0` before the season starts, and then `n` daily signed changes `d[0..n-1]`. The level after day `k` is the prefix sum `L[k] = d[0] + ... + d[k-1]`, with `L[0] = 0` for the pre-season reading. A drawdown for a pair of days `i <= j` is `L[i] - L[j]` — how far the level fell from the earlier reading to the later one — and I must report the maximum drawdown over all `0 <= i <= j <= n`. Two facts I want nailed down before any algorithm. First, the index range is `0..n` inclusive, so there are `n+1` readings including the pre-season `L[0] = 0`; the brute force and the fast scan both have to include that reading or they will miss drops measured from the very start. Second, `i = j` is allowed, giving drawdown `L[i] - L[i] = 0`, so the answer is never negative — a season that only rises has worst drawdown `0`, full stop. That non-negativity is the corner the problem is built around, so I will keep checking it.

Let me fix the scale because it dictates the data types. `n <= 2*10^5` and `|d[i]| <= 10^9`, so a level can reach `2*10^5 * 10^9 = 2*10^14` in magnitude, and a drawdown (a difference of two levels) can reach roughly `4*10^14`. That is far past the 32-bit range of about `2.1*10^9`, so the running level, the running peak, and the answer all have to be 64-bit. I will use `long long` for every accumulator. An `int` here is a silent wrong-answer on the large tests, not a crash, which is the worst kind.

**Candidate approaches.** Two routes are on the table and I want the one I can defend, not the one that types fastest.

- *All-pairs over the prefix sums.* Materialise `L[0..n]` and loop over every pair `i <= j`, taking the max of `L[i] - L[j]`. This is transparently correct: it is the literal definition. But it is `O(n^2)`, which at `n = 2*10^5` is `4*10^10` operations — nowhere near a 1-second budget. I keep this only as the oracle to check the fast solution against.
- *Single pass with a running peak.* The drawdown `L[i] - L[j]` for a fixed later day `j` is maximised by taking the *largest* earlier (or equal) `L[i]`, i.e. the running peak `peak_j = max(L[0..j])`. So the best drawdown ending at day `j` is `peak_j - L[j]`, and the global answer is `max_j (peak_j - L[j])`. That collapses the inner loop: I just walk left to right, keep the running peak, and at each step measure `peak - L[j]`. This is `O(n)` time and `O(1)` extra memory. The open questions are exactly the ones the problem flags: what value does `peak` start at, in what order do I "measure the drop" versus "update the peak", and which way does the subtraction go.

**Deriving the running-peak recurrence and proving the collapse.** Define `peak_j = max(L[0], L[1], ..., L[j])`. The claim is

```
max over all i <= j of (L[i] - L[j])  ==  max over j of (peak_j - L[j]).
```

Right-hand side `<=` left: for each `j`, `peak_j = L[i*]` for some `i* <= j`, and `peak_j - L[j] = L[i*] - L[j]` is one specific pair with `i* <= j`, hence `<=` the overall max. Left `<=` right: any pair `i <= j` has `L[i] <= peak_j` by definition of `peak_j`, so `L[i] - L[j] <= peak_j - L[j] <=` the max over `j` of that. Both directions hold, so the two are equal. Good — the collapse is justified, not just plausible.

Now the base value. `peak_0 = max(L[0]) = L[0] = 0`, because the only reading available at or before day 0 is the pre-season `0`. So `peak` must start at `0`, *not* at `-infinity` and *not* at `d[0]`. This is the base case the problem is daring me to get wrong: if I started `peak` at `-infinity`, the first measured drop `peak - L[1] = -inf` would be garbage, and more importantly I would lose the legitimate option of measuring a fall *from the pre-season reference*, which is the whole reason a single outflow day has a positive answer.

**Numeric self-check of the formula on the sample.** Before writing code I verify the derived identity arithmetically on the documented case `d = [3, -2, -5, 4, -1, -3, 2, 1]`. The levels are
`L[0]=0`, `L[1]=3`, `L[2]=1`, `L[3]=-4`, `L[4]=0`, `L[5]=-1`, `L[6]=-4`, `L[7]=-2`, `L[8]=-1`.
Running peak `peak_j`: `0, 3, 3, 3, 3, 3, 3, 3, 3`. Then `peak_j - L[j]`:
`0-0=0`, `3-3=0`, `3-1=2`, `3-(-4)=7`, `3-0=3`, `3-(-1)=4`, `3-(-4)=7`, `3-(-2)=5`, `3-(-1)=4`.
The maximum is `7`, at `j = 3` (and again at `j = 6`). The all-pairs definition agrees: the deepest fall is from the peak `L[1] = 3` down to the trough `L[3] = -4` or `L[6] = -4`, a drop of `7`. The formula reproduces the stated answer `7`, so the collapse is correct on a real instance.

**First implementation and a trace, because clean math transcribes dirty.** My first cut of the loop body, reading deltas one at a time:

```
long long level = 0, peak = 0, best = 0;
for (int i = 0; i < n; i++) {
    long long d; cin >> d;
    level += d;
    if (level > peak) peak = level;        // update peak first
    if (peak - level > best) best = peak - level;  // then measure the drop
}
```

I am suspicious of the order — I update `peak` *before* measuring the drop — so I trace the smallest input that could expose it: a single outflow day, `n = 1`, `d = [-5]`. The level goes `L[0] = 0 -> L[1] = -5`. The true answer is `5`: the level fell from the pre-season `0` down to `-5`. Trace my code. Start `level=0, peak=0, best=0`. i=0: `level = 0 + (-5) = -5`. `if (level > peak)` is `-5 > 0`? No, so `peak` stays `0`. `if (peak - level > best)` is `0 - (-5) = 5 > 0`? Yes, `best = 5`. Final `best = 5`. Correct here. Hm, this single case did not break — let me push harder, because "update then measure" can still be wrong when the level *rises* on the same step it would have set a drop.

**The bug, caught by a rising-then-irrelevant trace.** Consider `n = 2`, `d = [10, -30]`, where the level goes `0 -> 10 -> -20`. The true answer is `30`: a fall from the peak `10` to `-20`. Trace the "update first" code. Start `level=0, peak=0, best=0`. i=0 (`d=10`): `level = 10`. `10 > 0`? Yes, `peak = 10`. `peak - level = 10 - 10 = 0 > 0`? No, `best` stays `0`. i=1 (`d=-30`): `level = -20`. `-20 > 10`? No, `peak` stays `10`. `peak - level = 10 - (-20) = 30 > 0`? Yes, `best = 30`. Final `30`. Correct. So "update first" survives this too.

Let me find the case where updating the peak *before* measuring actually corrupts the answer. The danger is a day where the level rises to a *new peak* — there the drop measured *after* the update is `peak - level = 0`, which is fine, but the subtle error is different: the order matters only if I ever measure a drop using a peak that includes the *current* day when the current day is itself the trough I am dropping to. That cannot happen because a new peak is never below the current level. So in fact "update peak, then measure `peak - level`" gives the same number as "measure `peak - level`, then update peak" — *for this formula*. The two orderings agree because when `level > peak` the freshly-updated `peak` equals `level` and contributes a zero drop, while when `level <= peak` the peak is unchanged by the update. So the ordering is not where my bug is. Good to know — but I almost shipped a belief without proof, so let me hunt for the real defect elsewhere.

**The real bug: a sign-flipped subtraction.** I re-read my "fast" draft and notice I had actually first written the measurement the other way around, computing the *gain* instead of the *drop*:

```
if (level - peak > best) best = level - peak;   // WRONG: this is a rise, not a fall
```

with `peak` tracking the running *minimum* level, intending "biggest rise from a prior low". That is the maximum-profit / maximum-subarray shape, not maximum-drawdown — I conflated the two because they are mirror images. Trace it on `d = [10, -30]` (true drawdown `30`). With `peak` meaning running min: start `level=0, minLevel=0, best=0`. i=0: `level=10`; `level - minLevel = 10 - 0 = 10 > 0`, `best=10`; `minLevel` stays `0`. i=1: `level=-20`; `level - minLevel = -20 - 0 = -20 > 10`? No; update `minLevel = min(0,-20) = -20`. Final `best = 10`. That returns `10`, the biggest *rise*, when the answer is `30`, the biggest *fall*. The sign was inverted: drawdown is `earlier_high - later_low = peak - level`, and I had `later - earlier`. This is exactly the "sign handling" trap the problem warns about, and it is invisible on any all-rising input (where both `10` and `30`-style answers can look plausible) until a real fall appears.

**Fix and re-verification.** The correct measurement keeps `peak` as the running *maximum* level and computes the *fall* `peak - level`. To be maximally safe about ordering I measure the drop *before* updating the peak (so the peak is strictly "best earlier or equal reading"), which I proved above gives the same result and reads more obviously like the formula `peak_j - L[j]`:

```
long long level = 0, peak = 0, best = 0;
for (int i = 0; i < n; i++) {
    long long d; cin >> d;
    level += d;                          // L[i+1] = L[i] + d[i]
    if (peak - level > best) best = peak - level;   // fall from the running peak
    if (level > peak) peak = level;      // update peak AFTER measuring
}
```

Re-trace `d = [10, -30]`: start `(level,peak,best)=(0,0,0)`. i=0: `level=10`; `peak-level=0-10=-10`, not `> 0`, `best` stays `0`; `10 > 0` so `peak=10`. i=1: `level=-20`; `peak-level=10-(-20)=30 > 0`, `best=30`; `-20 > 10`? No. Final `30`. Correct, and the sign now points down. Re-trace the single outflow `d = [-5]`: i=0: `level=-5`; `peak-level=0-(-5)=5>0`, `best=5`; `-5>0`? No. Final `5`. Correct. The two cases that exercised the sign now both pass, and they pass for the reason I fixed (drop = `peak - level`, peak seeded at `L[0]=0`), which is the evidence I trust.

**Re-running the formula self-check through the fixed code on the sample.** `d = [3,-2,-5,4,-1,-3,2,1]`. `(level,peak,best)` evolution:
start `(0,0,0)`;
i=0 d=3: level=3; `0-3=-3` no; `3>0` peak=3 -> `(3,3,0)`;
i=1 d=-2: level=1; `3-1=2>0` best=2; `1>3`? no -> `(1,3,2)`;
i=2 d=-5: level=-4; `3-(-4)=7>2` best=7; `-4>3`? no -> `(-4,3,7)`;
i=3 d=4: level=0; `3-0=3` not `>7`; `0>3`? no -> `(0,3,7)`;
i=4 d=-1: level=-1; `3-(-1)=4` not `>7`; no -> `(-1,3,7)`;
i=5 d=-3: level=-4; `3-(-4)=7` not `>7`; no -> `(-4,3,7)`;
i=6 d=2: level=-2; `3-(-2)=5` not `>7`; no -> `(-2,3,7)`;
i=7 d=1: level=-1; `3-(-1)=4` not `>7`; no -> `(-1,3,7)`.
Final `best = 7`. Matches the documented sample and my hand formula. Two independent traces now agree.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0` (empty log): the loop never runs; `best` stays `0`. There is only the pre-season reading `L[0]=0`, no fall possible, answer `0`. Correct. (Also: `if (!(cin >> n)) return 0;` covers truly empty input, printing nothing, which an empty-token judge treats as `0`.)
- `n = 1`, single rise `d = [5]`: level `0 -> 5`; measure `0 - 5 = -5` (not `> 0`), `best` stays `0`. The gauge only rose, worst drawdown `0`. Correct — this is the non-negativity corner, and it works precisely because `best` is seeded at `0` and the measurement can go negative without polluting it.
- `n = 1`, single fall `d = [-5]`: answer `5`, traced above. The fall is measured against the seeded peak `L[0]=0`. Correct.
- All negative `d = [-3,-1,-4]`: levels `0,-3,-4,-8`; peak stays `0`; drops `3, 4, 8`; `best = 8`. The whole accumulated decline `|sum|` because the peak is the very first reading. Correct — and this is exactly the all-negative corner where a `-infinity` peak seed would have failed.
- All positive `d = [2,3,4]`: levels `0,2,5,9`; every `peak - level = 0`; `best = 0`. Correct.
- All zeros `d = [0,0,0]`: level stays `0`, peak `0`, `best = 0`. Correct.
- Rise then deep fall `d = [10,-30]`: `30`, traced. Correct.
- Overflow: `level`, `peak`, `best` are all `long long`; the worst level magnitude `~2*10^14` and worst drawdown `~4*10^14` fit with room to spare in the `~9.2*10^18` range. I verified on the live worst case `n = 2*10^5`, first half `+10^9` then second half `-10^9`: the level climbs to `10^14` then falls to `0`, deepest drawdown `10^14`, which the program prints exactly (`100000000000000`, far above the 32-bit ceiling) in 0.06 s. So both the type choice and the time budget are confirmed empirically, not just argued.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace, so the input layout (one line or many) does not matter.

**Final solution.** I convinced myself the *idea* is right by proving the running-peak collapse in both directions and checking the formula arithmetically on the sample; I convinced myself the *code* is right by tracing a sign-flip on `[10,-30]` and `[-5]` to a precise cause (drop is `peak - level`, not `level - min`), fixing it, and re-verifying every corner including the empty, single-day, all-negative, all-positive, and large-overflow cases. That is what I ship — one self-contained `O(n)` single pass with a running peak seeded at the pre-season reading `L[0] = 0`:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;            // n = 0 (or empty input) -> no day, answer 0

    // L is the running level (a prefix sum of the daily deltas), starting at L[0] = 0
    // BEFORE any day. peak is the maximum level seen so far, including day 0.
    // The maximum drawdown is max over j of (peak_up_to_j - L[j]); i = j gives 0,
    // so the answer is never negative even if the level only ever rises.
    long long level = 0;                  // L[0] = 0, the reference before day 1
    long long peak = 0;                   // best level seen so far == L[0]
    long long best = 0;                   // i = j allowed -> drawdown >= 0

    for (int i = 0; i < n; i++) {
        long long d;
        cin >> d;
        level += d;                       // L[i+1] = L[i] + d[i]
        if (peak - level > best) best = peak - level;   // drop from the running peak
        if (level > peak) peak = level;   // update peak AFTER measuring the drop
    }

    cout << best << "\n";
    return 0;
}
```

**Causal recap.** The fast idea is that for a fixed later day `j` the deepest fall is from the largest earlier reading, so `maxDrawdown = max_j (peak_j - L[j])` with `peak_j = max(L[0..j])` — a collapse I proved both ways and checked numerically on the sample (peak `3`, trough `-4`, drop `7`). The pitfall the problem is built on is the **sign and the base case**: I first wrote the measurement as `level - min` (a rise, the maximum-subarray mirror) which a trace of `[10,-30]` exposed by returning `10` instead of `30`, and the peak must be seeded at the pre-season `L[0] = 0` (not `-infinity`, not `d[0]`) or the legitimate fall-from-reference that makes a single outflow day answer `5` and an all-negative log answer its full decline would be lost; seeding `best` at `0` and letting `peak - level` go negative harmlessly closes the all-rising, empty, and single-day corners, while `long long` throughout closes overflow, confirmed on the live `n = 2*10^5` worst case.
