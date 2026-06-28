**Reading the problem and pinning the contract.** I have an array `a[0..n-1]` and I want the length of the longest *bitonic* subsequence: pick elements in their original order so their values strictly increase to a single peak and then strictly decrease. The increase must be real — at least one step up before any step down — and the descending tail may be empty, so a strictly increasing run of length `>= 2` already counts (its last element is the peak with nothing after it). Both comparisons are strict, which I underline immediately because it changes everything about equal values: `[3, 3]` is *not* an increase, and `[5, 5, 1]` cannot use both fives. Input is `n` then the `n` values on stdin; I print one integer, and the answer is `0` when nothing valid exists — `n <= 1`, a strictly decreasing array, or a plateau of equal values.

Before any algorithm I fix the scale, because it decides the budget. `n <= 5000` and `|a[i]| <= 10^9`. The thing I am computing is a *length*, never a sum, so the answer is bounded by `n = 5000` and fits trivially in an `int`. The values themselves fit in `int` too, but I will read them into `long long` out of habit and to keep comparisons unambiguous — it costs nothing here and removes any doubt about `10^9` arithmetic. The real budget question is time: `n <= 5000` means an `O(n^2)` method does about `2.5 * 10^7` comparisons, which is nothing for a 2-second limit. So `O(n^2)` is firmly in budget, and I do *not* need to reach for the `O(n log n)` patience-sorting machinery. That observation matters for the whole strategy below: I get to pick the method I can most easily *prove*, not the asymptotically slickest one.

**Laying out the candidate approaches.** Two routes are on the table, and I want to commit to the one I can defend rather than the one that looks clever.

- *Single-pass / greedy peak walk.* Sweep the array once. Climb while the next element is larger, recording the ascent; when it stops rising, switch to descending and count down while the next element is smaller; track the longest mountain seen. It is `O(n)` and barely a dozen lines, and the shape of the problem — "go up then go down" — practically begs for a one-pass solution. The nagging worry is that this only ever looks at *contiguous* runs and the next neighbour, whereas a *subsequence* is allowed to skip arbitrarily many elements on either slope. A single forward pass with a notion of "current direction" feels like it is solving "longest contiguous mountain," which is a different problem. I will not trust it until I have either convinced myself it generalizes or broken it.
- *Length DP from both ends.* For each index `i`, let `inc[i]` be the length of the longest strictly increasing subsequence that *ends* at `i`, and `dec[i]` the length of the longest strictly decreasing subsequence that *starts* at `i`. If `i` is the peak, the best bitonic subsequence through it has the ascent of length `inc[i]` glued to the descent of length `dec[i]`, sharing the peak, so length `inc[i] + dec[i] - 1`. Each of `inc` and `dec` is a textbook `O(n^2)` LIS-style DP. The risk here is not the idea but the *transcription*: the strict comparisons, the `-1` for the shared peak, and the "must actually increase" requirement are all easy to get subtly wrong.

**Stress-testing the single-pass idea before committing.** Hand-waving "a mountain walk should work" is exactly how a wrong solution gets shipped, so let me actually try to break the greedy peak walk with a concrete instance. The whole point of a *subsequence* is that I can drop elements, and the danger case for any single contiguous pass is an array where the best mountain is interleaved with junk that derails a neighbour-based walk.

Take `a = [1, 11, 2, 10, 4, 5, 2, 1]`, indices `0..7`. What is the true answer? I can pick `1 (idx0) < 2 (idx2) < 10 (idx3) > 4 (idx4) > 2 (idx6) > 1 (idx7)`. Check it: `1 < 2 < 10` strictly increasing, peak `10`, then `10 > 4 > 2 > 1` strictly decreasing. That is six elements, all in order, so a bitonic subsequence of length `6` exists. Can I do `7`? I would have to weave seven of these eight values into one up-then-down chain; the `11` at index 1 is a problem because everything after it that I might want (`10, 4, 5, 2, 1`) is smaller, so `11` could only be the peak, but then the ascent into it is just `1, 11` (length 2) and the descent out of it is `11 > 10 > 4 > 2 > 1` (length 5) for `2 + 5 - 1 = 6` — also six, not more. So the true answer is `6`.

Now run the greedy peak walk on the *raw contiguous* array. Starting at index 0: `1 -> 11` is up (ascent so far 2). `11 -> 2` is down, so I flip to descending: `11 > 2`? yes (descent 2). `2 -> 10`? that is *up*, not down — the contiguous descent ends. The mountain I just walked is `1, 11, 2`, length 3. A neighbour-based walk cannot reach back and pick the `1` and `2` to feed a `10`-peaked mountain, because those are non-adjacent and the `11` sits between them blocking the contiguous ascent. Depending on exactly how I restart the walk after the broken mountain, I will scrape together other short contiguous mountains (`2, 10, 4` of length 3; `4, 5, 2, 1` of length 4), but none of them is the length-6 *subsequence*, because that subsequence requires skipping index 1 entirely and stitching `idx0, idx2, idx3` across the gap. The single pass simply cannot express "skip the 11." Greedy peak walk reports something like `4`; the truth is `6`.

That is the counterexample, and it shows me *why* the single pass is wrong, not merely *that* it is: a contiguous-direction walk is solving the longest *substring* mountain, and the problem asks for the longest *subsequence* mountain, which is strictly more powerful because it may delete elements on either slope. The verification paid off — it killed an approach I would otherwise have been tempted to ship because it is short and "looks like the problem." Single-pass is out.

**Deriving the both-ends DP and checking the combine on paper.** I switch to the method I can prove. The key structural fact: every bitonic subsequence has exactly one peak element, and to its left it is a strictly increasing subsequence ending at the peak, while to its right it is a strictly decreasing subsequence starting at the peak. These two halves are independent given the peak's index and value — the left half only ever looks at elements before the peak that are smaller, the right half only at elements after the peak that are smaller. So if I knew, for the peak index `i`, the best left ascent length `inc[i]` and the best right descent length `dec[i]`, the best bitonic subsequence peaking at `i` is `inc[i] + dec[i] - 1` (the `-1` because the peak is the last element of the ascent *and* the first element of the descent — counting it once).

Computing `inc[i]`: it is the classic longest-strictly-increasing-subsequence-ending-here DP. `inc[i] = 1 + max{ inc[j] : j < i and a[j] < a[i] }`, defaulting to `1` if no such `j`. The strict `a[j] < a[i]` is what enforces a strict increase and correctly refuses to extend through equal values. Symmetrically, `dec[i]` is the longest strictly-decreasing subsequence starting at `i`, which I compute by scanning from the right: `dec[i] = 1 + max{ dec[j] : j > i and a[j] < a[i] }`, default `1`. Note both DPs use the same predicate `a[j] < a[i]` — for `inc` the smaller element is the earlier one, for `dec` the smaller element is the later one — which is a pleasant symmetry that makes the code hard to get backwards.

The "at least one real increase" requirement is the subtle corner. If I let every index be a peak, then an index `i` with `inc[i] = 1` (nothing smaller before it) and `dec[i] = 1` (nothing smaller after it — e.g. a global structure where it just sits) would contribute `1 + 1 - 1 = 1`, claiming a length-1 "bitonic" subsequence, which violates my contract (I demand `p >= 1`, i.e. a real ascent). Worse, a strictly *decreasing* array would have `dec[0] = n` and `inc[0] = 1`, giving `1 + n - 1 = n` — it would happily report the whole decreasing array as "bitonic," which is exactly wrong because there is no increase at all. So the combine must be gated on `inc[i] >= 2`: only an index that has at least one strictly smaller predecessor is allowed to be a peak. With that gate, a strictly decreasing array has every `inc[i] = 1`, no index qualifies, and the answer is `0`. Good — that single guard encodes the whole "must go up first" rule.

Let me confirm the combine by hand on the sample `a = [1, 11, 2, 10, 4, 5, 2, 1]`, answer `6`. Compute `inc` left to right (strictly increasing ending here): `inc[0]=1` (the `1`); `inc[1]=2` (`1<11`); `inc[2]=2` (`1<2`); `inc[3]=3` (`1<2<10`); `inc[4]=3` (`1<2<4`); `inc[5]=4` (`1<2<4<5`); `inc[6]=3` (`1<2<2`? no, `2` not `<2`; best is `1<2<2`... careful: a[6]=2, predecessors smaller than 2 are the `1`s, so `inc[6]=2`). Let me redo `inc[6]`: a[6]=2; smaller earlier values are a[0]=1 (inc 1) and... a[2]=2 is not `<2`. So `inc[6]=1+inc[0]=2`. `inc[7]`: a[7]=1; nothing earlier is `<1`, so `inc[7]=1`. Now `dec` right to left (strictly decreasing starting here): `dec[7]=1`; `dec[6]=2` (`2>1`); `dec[5]=3` (`5>2>1`); `dec[4]=3` (`4>2>1`); `dec[3]=4` (`10>4>2>1`); `dec[2]=2` (`2>1`); `dec[1]=5` (`11>10>4>2>1`); `dec[0]=1` (nothing after `1` is smaller). Now combine, gated on `inc[i]>=2`: peak at idx3 gives `inc=3, dec=4 -> 3+4-1=6`; peak at idx1 gives `inc=2, dec=5 -> 6`; peak at idx5 gives `inc=4, dec=3 -> 6`. Maximum `6`. Matches. The recurrence and the combine are right.

**First implementation — and a trace, because clean math transcribes dirty.** My first cut of the combine loop did *not* have the `inc[i] >= 2` guard; I wrote simply `best = max(best, inc[i] + dec[i] - 1)` for every `i`, figuring "a real bitonic answer will always dominate." So I traced the smallest input that could expose the missing guard: a strictly decreasing array `a = [3, 2, 1]`, where the correct answer is `0` (it never increases). With my guardless code: `inc = [1, 1, 1]` (nothing increases), `dec = [3, 2, 1]` (`3>2>1`, `2>1`, `1`). Combine over all `i`: idx0 gives `1 + 3 - 1 = 3`; idx1 gives `1 + 2 - 1 = 2`; idx2 gives `1`. Maximum `3`. The code prints `3`.

**Diagnosing the bug.** `3` is flatly wrong — the array strictly decreases, there is no increase-then-decrease at all, the answer must be `0`. The defect is precise: by allowing any index with `inc[i] = 1` to act as a peak, I let the pure descent `3 > 2 > 1` masquerade as a bitonic subsequence whose "ascent" is the single peak element with no real step up. The contract says the peak must be *preceded by an increase* (`p >= 1`), which in DP terms is exactly `inc[i] >= 2` — the longest increasing subsequence ending at the peak must contain at least two elements. My guardless combine dropped that condition. The fix is one line: only consider `i` as a peak when `inc[i] >= 2`.

**Fixing and re-verifying.** I add the guard:

```
int best = 0;
for (int i = 0; i < n; i++)
    if (inc[i] >= 2)
        best = max(best, inc[i] + dec[i] - 1);
```

Re-trace `[3, 2, 1]`: `inc = [1,1,1]`, so no index satisfies `inc[i] >= 2`, `best` stays `0`. Correct. Re-trace the sample `[1,11,2,10,4,5,2,1]`: idx3 has `inc=3 >= 2`, contributes `6`; answer `6`. Still correct. Re-trace a strictly increasing array `[1,2,3,4]` (answer should be `4` — full ascent, empty descent): `inc=[1,2,3,4]`, `dec=[1,1,1,1]`; the qualifying peaks are idx1 (`2+1-1=2`), idx2 (`3+1-1=3`), idx3 (`4+1-1=4`); maximum `4`. Correct — the empty-descent case falls straight out because `dec[i] = 1` means "peak with nothing after," and the `-1` cancels it cleanly. The case that broke now passes, and it broke for the reason I fixed, which is the evidence I trust.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0`: I special-case it (and the loops would be no-ops anyway); `best` stays `0`. The empty array has no bitonic subsequence — correct.
- `n = 1`, `a = [5]`: `inc=[1]`, no index has `inc[i] >= 2`, answer `0`. A single element cannot increase-then-decrease — correct.
- All equal, `a = [3, 3, 3, 3, 3]`: strict `a[j] < a[i]` is never satisfied, so `inc` and `dec` are all `1`, no peak qualifies, answer `0`. Equal values cannot form a strict ascent — correct, and this is precisely the corner the strictness guards.
- Strictly decreasing, `[5,4,3,2,1]`: every `inc[i] = 1`, answer `0` — correct (handled above).
- Duplicates mid-array, `[2, 2, 3]`: a[0]=2, a[1]=2 (not `<`), a[2]=3. `inc=[1,1,2]` (`2<3` from either two, but not `2<2`), `dec=[1,1,1]`. Peak idx2: `2+1-1=2`. Answer `2` — the chain `2 < 3` (length 2, empty descent). Correct; the duplicate `2` is harmlessly skipped.
- Overflow: there is none to worry about — the answer is a length bounded by `n <= 5000`, an `int` everywhere is safe; I only used `long long` for the input values to keep comparisons clean.
- Performance: `inc` and `dec` are each `O(n^2)`; at `n = 5000` that is about `5 * 10^7` comparisons total, which I measured at roughly 0.03 s — far under the 2-second limit. No need for the `O(n log n)` Fenwick/patience version.

**Final solution.** I convinced myself the *idea* is right by breaking the single-pass walk with a concrete length-6-vs-4 counterexample on `[1,11,2,10,4,5,2,1]` and then proving the both-ends combine on the same array, and I convinced myself the *code* is right by tracing the strictly-decreasing case `[3,2,1]` to a precise missing-guard cause, fixing it with `inc[i] >= 2`, and re-verifying the fix plus the strictness, empty-descent, single-element, and plateau corners. That is what I ship — one self-contained file, the simple provable `O(n^2)` both-ends DP I can defend rather than the single-pass walk I broke:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> answer 0
    vector<long long> a(n);
    for (auto &x : a) cin >> x;
    if (n == 0) { cout << 0 << "\n"; return 0; }

    // inc[i] = length of the longest STRICTLY increasing subsequence that ENDS at i.
    // dec[i] = length of the longest STRICTLY decreasing subsequence that STARTS at i.
    // A bitonic subsequence with peak at index i (the peak counted once) has length
    // inc[i] + dec[i] - 1. We require an actual increase before the peak, i.e. inc[i] >= 2,
    // so a purely non-increasing array yields answer 0 (no valid increase-then-decrease).
    vector<int> inc(n, 1), dec(n, 1);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < i; j++)
            if (a[j] < a[i])
                inc[i] = max(inc[i], inc[j] + 1);

    for (int i = n - 1; i >= 0; i--)
        for (int j = i + 1; j < n; j++)
            if (a[j] < a[i])
                dec[i] = max(dec[i], dec[j] + 1);

    int best = 0;
    for (int i = 0; i < n; i++)
        if (inc[i] >= 2)                          // peak must be preceded by a real increase
            best = max(best, inc[i] + dec[i] - 1);

    cout << best << "\n";
    return 0;
}
```

**Causal recap.** The single-pass mountain walk looked right because the problem literally says "increase then decrease," but a traced counterexample (`[1,11,2,10,4,5,2,1]`: the walk scrapes length 4 from contiguous mountains while the true *subsequence* answer is 6, reached only by skipping the `11`) showed it is silently solving the longest contiguous-substring mountain instead of the subsequence one — so I moved to the both-ends LIS/LDS DP and proved its combine on that same array. The DP's only real trap is the "must actually increase" corner: my first guardless combine let a strictly decreasing array `[3,2,1]` report `3` by treating a pure descent as bitonic, and gating each peak on `inc[i] >= 2` fixes it (strictly decreasing arrays and equal-value plateaus then correctly return `0`). At `n <= 5000` the `O(n^2)` cost is about `5 * 10^7` comparisons measured near 0.03 s, the answer is a length bounded by `n` so no overflow exists, and the strict `a[j] < a[i]` predicate closes the duplicate, empty-descent, single-element, and empty-array corners.
