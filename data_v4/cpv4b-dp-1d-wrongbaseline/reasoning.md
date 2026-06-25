**Reading the problem and pinning the contract.** I have `n` slots on a closed loop, each with a signed profit `a[i]`, and I must pick one contiguous run of slots — which may wrap past the seam from `n-1` back to `0` — using each slot at most once, so the run length is between `1` and `n`. The run must be non-empty, and I maximize the shipped sum. Input is `n` then the `n` values; output is one integer. Before any algorithm I fix the scale, because it dictates the types: `n <= 2*10^5` and `|a[i]| <= 10^9`, so the sum of an entire loop can reach `2*10^5 * 10^9 = 2*10^14`. That is far beyond the 32-bit range of about `2.1*10^9`, so every accumulator — the running Kadane value, the total, the minimum — has to be 64-bit. I will use `long long` everywhere. An `int` here is a silent wrong answer on the large tests, so this is non-negotiable.

**The two structural cases of a circular run.** A non-empty circular contiguous run is one of exactly two shapes. Either it does *not* cross the seam, in which case it is an ordinary subarray `a[l..r]` with `0 <= l <= r <= n-1` — a plain non-circular contiguous segment. Or it *does* cross the seam, meaning it occupies a suffix `a[k..n-1]` glued to a prefix `a[0..m]`. Any run that uses all `n` slots can be viewed either way; I will keep that ambiguity in mind because it is where the corner lives. The maximum over all runs is the max of (best non-wrapping run) and (best wrapping run). This case split is the whole game.

**Candidate approaches.** Two routes are on the table, and I want the one I can *prove* on this exact "non-empty" contract, not just the one everybody quotes.

- *The textbook wrap formula.* The best non-wrapping run is plain Kadane: scan keeping `cur = max(a[i], cur + a[i])` and `best = max(best, cur)`. For the wrapping run, here is the slick trick everyone uses: a wrapping run is "the whole loop minus a contiguous chunk that I leave out", and the chunk I leave out is itself a non-wrapping subarray (the gap between the prefix and the suffix sits in the middle and does not wrap). To maximize `total - gap` I minimize `gap`, where `gap` ranges over non-wrapping subarrays. So compute `worst` = minimum non-empty subarray sum (Kadane with min/`+` flipped), and the wrapping candidate is `total - worst`. Final answer `max(best, total - worst)`. It is `O(n)`, two passes, and it looks airtight. The open question — the one I refuse to wave away — is whether `total - worst` is always a *legal* non-empty run under this contract.
- *Brute search.* Try every start `s` in `0..n-1` and every length `L` in `1..n`, summing around the circle; keep the max. `O(n^2)`, obviously correct, usable only as an oracle on small cases. I will use it to attack the textbook formula rather than to ship.

**Deriving the wrap formula carefully and self-checking it on a concrete case.** Let me not just quote `max(best, total - worst)` — let me verify the wrapping half on a case where the wrap genuinely wins, the documented sample `a = [5, -3, 5]`. Brute first, so I have ground truth. Runs and their circular sums: length 1: `5, -3, 5` -> best `5`. Length 2: `{0,1}=2`, `{1,2}=2`, `{2,0}=5+5=10`. Length 3: the whole loop `=7`. So the true answer is `10`, achieved by the wrapping run `{slot 2, slot 0}`. Now the formula. Kadane for `best`: start `cur=best=5`; i=1 `cur=max(-3,5-3)=2`, `best=5`; i=2 `cur=max(5,2+5)=7`, `best=7`. So `best=7` (the whole loop, which is non-wrapping when read as `a[0..2]`). `total = 5-3+5 = 7`. Min subarray `worst`: start `curMin=worst=5`; i=1 `curMin=min(-3,5-3)=-3`, `worst=-3`; i=2 `curMin=min(5,-3+5)=2`, `worst=-3`. So `worst=-3`, and `total - worst = 7 - (-3) = 10`. Answer `max(7, 10) = 10`. Matches the brute. Good — the wrapping arithmetic is sound when a real gap exists: dropping the single slot `{1}` (value `-3`, the minimum subarray) leaves `{2,0}` summing to `10`, exactly `total - worst`. The formula's logic is "leave out the worst contiguous chunk", and here it leaves out one slot, which is legal.

**First implementation — the textbook formula verbatim.** I write the obvious thing, two Kadane passes plus the wrap combine:

```
long long best = a[0], cur = a[0], total = a[0];
long long worst = a[0], curMin = a[0];
for (int i = 1; i < n; i++) {
    cur = max(a[i], cur + a[i]);     best  = max(best,  cur);
    curMin = min(a[i], curMin + a[i]); worst = min(worst, curMin);
    total += a[i];
}
long long answer = max(best, total - worst);   // textbook
```

This passed my `[5,-3,5]` trace by hand. But I have learned not to trust a formula I have only seen win — I need to see it on a case where the *wrap should not be taken at all*. The most suspicious family is all-negative belts, because there "leave out the worst chunk" might want to leave out *everything*.

**First debug episode — tracing the textbook formula on an all-negative belt.** Take `a = [-3, -1, -4]`. Brute ground truth first: every non-empty run is negative; the largest (least negative) single slot is `-1`, and any longer run is more negative (`{0,1}=-4`, `{1,2}=-5`, whole loop `-8`). So the true answer is `-1`. Now run the textbook formula. Kadane `best`: `cur=best=-3`; i=1 `cur=max(-1,-3-1)=-1`, `best=max(-3,-1)=-1`; i=2 `cur=max(-4,-1-4)=-4`, `best=max(-1,-4)=-1`. So `best=-1`. `total = -3-1-4 = -8`. Min subarray `worst`: `curMin=worst=-3`; i=1 `curMin=min(-1,-3-1)=-4`, `worst=min(-3,-4)=-4`; i=2 `curMin=min(-4,-4-4)=-8`, `worst=min(-4,-8)=-8`. So `worst=-8`. Then `total - worst = -8 - (-8) = 0`, and the textbook answer is `max(best, total - worst) = max(-1, 0) = 0`.

**The bug, pinned to its cause.** The formula returns `0`; the truth is `-1`. The defect is exact and it is *not* a transcription slip — the algorithm itself is wrong for this contract. Here `worst = -8` is the minimum subarray, and the minimum subarray is the *entire array* `a[0..2]`. So `total - worst` corresponds to "ship the loop minus the whole loop", i.e. the **empty run**. The empty run sums to `0`, which is what `total - worst` evaluated to. But this problem requires a non-empty run of length `1..n`; the empty selection is illegal. The textbook formula silently allows the wrap candidate to degenerate into the empty set whenever the minimum subarray swallows the whole array — which happens precisely when every element is non-positive in aggregate, the all-negative case. This is the famous trap in circular maximum-subarray, and my hand trace caught it producing a concretely wrong `0` instead of `-1`. I am glad I refused to ship on "the standard solution says so."

**The fix, derived from the cause.** The wrapping candidate `total - worst` is only legal when the gap I leave out is a *proper, non-empty* subarray — i.e. when `worst` is the sum of a subarray that is *not* the entire belt, so that at least one slot remains in the wrapping run. The exact signal that the minimum subarray is the whole belt is `worst == total`: the minimum non-empty subarray sum equals the sum of everything, which can only happen when the best (lowest) chunk to drop is literally all `n` slots. When that holds, the wrap option is illegal and I must not use it; the answer is just the best non-wrapping run `best` (which, being plain Kadane on a non-empty array, is always a legal non-empty run — the single least-negative element survives). So:

```
long long answer;
if (worst == total) answer = best;                 // wrap would be empty -> illegal
else                answer = max(best, total - worst);
```

Why is `worst == total` the right guard and not, say, "all elements negative"? Because it is the *direct* statement of the failure: the wrap is empty exactly when the dropped gap is the whole belt, i.e. `worst` (min subarray) equals `total`. Let me sanity-check the equivalence on the failing case: `worst=-8`, `total=-8`, equal -> take `best=-1`. Correct. And on `[5,-3,5]`: `worst=-3`, `total=7`, not equal -> `max(7, 10)=10`. Correct, the guard does not fire when it should not. One more, a mixed belt with a clear interior minimum, `a = [4, -1, 2, -7, 3]`: `total = 1`. Min subarray is `{3}=-7` (any extension only raises it: `{2,3}=-5`, `{3,4}=-4`), so `worst=-7`, not equal to `total=1`; wrap candidate `1-(-7)=8` = run `{4,0,2}` wrapping the seam (`3+4+(-1)+2`? wait that is `{4,0,1,2}` = `3+4-1+2=8`), yes a legal wrap, and `best` from Kadane is `{0,1,2}=5` or `{4,0,2,...}`; `max(5,8)=8`. The guard correctly stays off. The guard fires only when the entire belt is the minimum subarray.

**Second debug episode — tracing the single-slot belt, `n = 1`.** Edge cases on circular problems love to die at `n=1`, where wrapping and non-wrapping coincide, so I trace `a = [-7]`. Brute: the only run is `{0}`, sum `-7`; answer `-7`. My code: the `i`-loop from `1` never runs, so `best = a[0] = -7`, `total = a[0] = -7`, `worst = a[0] = -7`. Guard: `worst == total`? `-7 == -7`, yes -> `answer = best = -7`. Correct. Now the *positive* single slot, `a = [9]`: `best=9, total=9, worst=9`, guard `9==9` true -> `answer = 9`. Correct. Here the guard fires even though the belt is positive, and that is *fine*: with `n=1` the wrap candidate `total - worst = 0` is the empty run anyway, so taking `best` is exactly right. The guard's meaning ("don't let the wrap go empty") is robust at `n=1`, which reassures me the condition is the true invariant and not a patch tuned to the all-negative case.

**Re-verifying that the bug is fixed and the fix did not break the wins.** I re-run the two cases that mattered. `[-3,-1,-4]`: now `worst=-8 == total=-8`, so `answer = best = -1`. Fixed — it was `0`, now `-1`. `[5,-3,5]`: `worst=-3 != total=7`, so `answer = max(7, 10) = 10`. Still the correct wrap. The case that broke now passes, for exactly the reason I identified (the wrap was degenerating to empty), and the case that worked still works. That is the evidence I trust: the fix is causally tied to the defect, not a coincidental nudge.

**Edge cases, deliberately, because this is where circular DP dies.**
- *All-negative belt:* covered above — guard fires, answer is the least-negative single slot. Without the guard it would wrongly be `0`.
- *All-positive belt:* the best run is the whole loop. Kadane `best` accumulates everything = `total`; `worst` is the smallest single element (Kadane-min never extends because every prefix-extend only grows the sum), so `worst < total` (for `n>=2`), and `total - worst = total - (min element) > total`?? That would *exceed* the whole loop — but wait, that is impossible since the whole loop is the max. Let me check this carefully on `a=[1,2,3]`: `total=6`, `worst`=min subarray=`{0}=1`, so `total-worst=5`. `best`=Kadane=`6`. `max(6,5)=6`. Good — `total - worst` gives `5` (drop slot 0, ship `{1,2}=2+3=5`), which is *less* than the whole loop `6`, so `best` wins. My worry was unfounded: `total - worst` drops a *positive* chunk so it is smaller than `total`, and `best` already includes the whole loop. Answer `6`, correct.
- *Belt with zeros,* e.g. `a=[0,0,0]`: `best=0`, `total=0`, `worst=0`, guard `0==0` true -> `answer=best=0`. The brute also gives `0` (any single zero slot). Correct, and notably the guard fires here too, which is harmless because `total-worst=0=best` anyway.
- *`n = 0`:* the contract says `n >= 1`, but I guard it: if `n==0` print `0` and return, so I never index `a[0]` out of bounds. Defensive, costs nothing.
- *Overflow:* `total` of an all-`10^9` belt is `2*10^14`, fits in `long long` with room to spare; `cur`, `curMin`, `worst`, `best` are all bounded by `+-total` in magnitude, so 64-bit is safe throughout. No sentinel arithmetic anywhere that could underflow.
- *Output:* exactly one integer and a newline; `cin >>` consumes arbitrary whitespace so the parsing is format-agnostic.

**Numeric self-check of the overflow claim.** I assert `total` fits in 64-bit. The signed `long long` max is `9223372036854775807 ~ 9.2*10^18`. My worst `|total| = 2*10^5 * 10^9 = 2*10^14`, and `2*10^14 < 9.2*10^18` by more than four orders of magnitude. `total - worst` in the extreme is at most `total - (-total) = 2*total = 4*10^14`, still `< 9.2*10^18`. So no accumulator overflows. Confirmed numerically, not asserted.

**Complexity.** Two interleaved Kadane scans in one pass: `O(n)` time, `O(1)` extra state beyond the input array. At `n = 2*10^5` this is trivially within a 1-second limit (a single linear pass over a few hundred thousand `long long`s).

**Final solution.** I convinced myself the *idea* needs guarding by tracing the textbook formula to a concrete wrong `0` on `[-3,-1,-4]`, I located the cause (the wrap candidate degenerating into the empty run when the minimum subarray is the whole belt, i.e. `worst == total`), I fixed it with that exact guard, and I re-verified the failing case flips to `-1` while the wrapping win `[5,-3,5]=10` and the `n=1` corners stay correct. That is what I ship — one self-contained file, the guarded circular-maximum DP I can defend rather than the bare textbook formula I broke:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;
    if (n == 0) { cout << 0 << "\n"; return 0; }  // contract guarantees n>=1, but be safe

    // Linear (non-wrapping) maximum subarray sum, non-empty (Kadane).
    long long best = a[0], cur = a[0];
    long long total = a[0];
    // Linear minimum subarray sum, non-empty (Kadane on negated logic).
    long long worst = a[0], curMin = a[0];
    for (int i = 1; i < n; i++) {
        cur = max(a[i], cur + a[i]);
        best = max(best, cur);
        curMin = min(a[i], curMin + a[i]);
        worst = min(worst, curMin);
        total += a[i];
    }

    // Wrapping candidate: total minus the minimum interior subarray.
    // If worst == total then every element is in the minimum subarray, i.e.
    // the "complement" wrap would be empty; that is illegal (segment must be
    // non-empty), so we must NOT take the wrap in that case.
    long long answer;
    if (worst == total) {
        // All elements lie in the minimum subarray => array is all non-positive
        // in the sense that the best non-empty pick is just the linear best.
        answer = best;
    } else {
        answer = max(best, total - worst);
    }

    cout << answer << "\n";
    return 0;
}
```

**Causal recap.** The textbook circular-maximum trick `max(Kadane, total - minSubarray)` looked airtight and even passed my `[5,-3,5]=10` wrap check, but tracing it on the all-negative belt `[-3,-1,-4]` produced a concretely wrong `0` instead of `-1`; the cause is precise — when the minimum subarray swallows the whole belt (`worst == total`), `total - worst` represents shipping the *empty* run, which this contract forbids — so I guard with `if (worst == total) answer = best`, which falls back to the always-legal non-empty Kadane result; re-verification flips the broken case to `-1` while preserving the wrap win and the `n=1`, all-positive, zeros, and overflow corners, all on 64-bit accumulators that comfortably hold the `2*10^14` worst-case total.
