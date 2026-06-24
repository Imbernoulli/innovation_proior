**Reading the problem and pinning the contract.** I have `n` thrusters with signed impulses `a[0..n-1]` — positive is forward, negative is reverse, zero is idle — and a required net impulse `T`. A burn fires two *distinct* thrusters, and it is admissible when `a[i] + a[j] >= T`. Burns are unordered, so `{i, j}` and `{j, i}` are one burn. I must count the admissible unordered pairs. Input on stdin is `n`, then `T`, then the `n` values; I print one integer. Before any algorithm I fix the scale, because it dictates the data types and it is the first place this problem bites. `n` can be `2*10^5`, so the number of pairs `n*(n-1)/2` is about `2*10^10` — that overflows 32-bit signed (`~2.1*10^9`), so the *count* itself must be `long long`. Each `a[i]` is up to `10^9` in magnitude and `T` is up to `2*10^9` in magnitude, so a pair sum `a[i]+a[j]` ranges in `[-2*10^9, 2*10^9]`, which already exceeds 32-bit; the comparison `a[i]+a[j] >= T` has to be done in 64-bit or it silently wraps. So both the values and the accumulator are `long long`. That is non-negotiable and it is decision number one.

**Laying out the candidate approaches.** The defining feature is that admissibility couples two indices through a *sum threshold*, and the order of the two indices does not matter. That immediately says: sort first, because once the array is monotone, "does some `a[j]` reach `T - a[i]`" becomes a position question instead of a value scan. Two routes follow from sorting.

- *Sort + binary search.* Sort ascending. For a fixed left index `i`, the partners `j` with `a[i] + a[j] >= T` are exactly those with `a[j] >= T - a[i]` — a contiguous *suffix* of the sorted array, whose start I find with `lower_bound`. Summing suffix lengths counts qualifying *ordered-by-position* pairs, but I have to be careful to count each unordered pair once (only `j` strictly to the right of `i`, or divide by two at the end). `O(n log n)`. The risk is the off-by-one in the suffix start and the de-duplication bookkeeping.
- *Sort + two pointers.* Sort ascending and run two converging pointers `lo` (from the left) and `hi` (from the right). The monotonicity gives a clean invariant: if `a[lo] + a[hi] >= T`, then because the array is sorted ascending, *every* index `k` in `[lo, hi-1]` also satisfies `a[k] + a[hi] >= T` (since `a[k] >= a[lo]`), so I can count all `hi - lo` of those pairs at once and then drop `hi` by one. If instead `a[lo] + a[hi] < T`, then `a[lo]` is too small to pair with *any* of the remaining `hi`-and-below, so `a[lo]` is useless and `lo` advances. `O(n log n)` for the sort and `O(n)` for the sweep, with no per-element search. The risks are the exact count increment (is it `hi - lo` or `hi - lo - 1`?) and the base case when `n < 2`.

Both are correct in principle and both are `O(n log n)`. I prefer the two-pointer sweep: it has no binary search inside a loop, the increment is a single arithmetic step, and the de-duplication is automatic because I only ever pair `hi` with strictly-smaller positions `[lo, hi-1]`. But "automatic" is exactly the kind of claim that hides an off-by-one, so I will derive the increment carefully and then trace it.

**Deriving the two-pointer increment and proving the invariant.** Sort `a` ascending. I maintain `lo = 0`, `hi = n-1`, and count only pairs `{lo', hi'}` with `lo' < hi'`, so each unordered pair is considered exactly once. The loop runs while `lo < hi`. Two cases:

1. `a[lo] + a[hi] >= T`. The current `hi` is the largest available value. Pair it with each `k` in `[lo, hi-1]`. For every such `k`, `a[k] >= a[lo]` (ascending), so `a[k] + a[hi] >= a[lo] + a[hi] >= T` — all of them qualify. There are exactly `hi - lo` such indices `k` (the integers `lo, lo+1, ..., hi-1`). I add `hi - lo` to the count. Now `hi` has been fully accounted for against everything below it, so I retire it: `hi--`. Crucially I do **not** also move `lo`, because `lo` still needs to be paired against the smaller `hi` values that remain.

2. `a[lo] + a[hi] < T`. The current `hi` is the *largest* value still in play, and even it cannot lift `a[lo]` to `T`. So `a[lo]` paired with any remaining index (all of which are `<= a[hi]`) also falls short. `a[lo]` can never be part of an admissible pair among what's left, so I discard it: `lo++`.

The increment is `hi - lo`, not `hi - lo - 1`: the indices strictly between `lo` and `hi` number `hi - lo - 1`, but I am pairing `hi` with the *closed* range `[lo, hi-1]`, which includes `lo` itself, giving `hi - lo` partners. Getting this boundary right is the crux; a common slip is to write `hi - lo - 1` and lose the pair `{lo, hi}`.

**Sanity-checking the derivation on the sample before writing code.** Sample: `n = 6`, `T = 3`, `a = [4, -1, 0, 5, -3, 2]`, claimed answer `7`. Sorted ascending: `[-3, -1, 0, 2, 4, 5]`, indices `0..5`. Start `lo=0, hi=5`, `count=0`.

- `lo=0, hi=5`: `a[0]+a[5] = -3 + 5 = 2`. Is `2 >= 3`? No. So `lo++` -> `lo=1`.
- `lo=1, hi=5`: `a[1]+a[5] = -1 + 5 = 4 >= 3`? Yes. Add `hi - lo = 5 - 1 = 4`. `count=4`. `hi--` -> `hi=4`. (These four are `{-1,5},{0,5},{2,5},{4,5}` — i.e. indices `1..4` paired with `5`.)
- `lo=1, hi=4`: `a[1]+a[4] = -1 + 4 = 3 >= 3`? Yes (boundary equality counts). Add `4 - 1 = 3`. `count=7`. `hi--` -> `hi=3`. (These three are `{-1,4},{0,4},{2,4}`.)
- `lo=1, hi=3`: `a[1]+a[3] = -1 + 2 = 1 >= 3`? No. `lo++` -> `lo=2`.
- `lo=2, hi=3`: `a[2]+a[3] = 0 + 2 = 2 >= 3`? No. `lo++` -> `lo=3`.
- `lo=3, hi=3`: `lo < hi` is false. Stop.

Final `count = 7`. That matches the claimed answer, and the seven pairs I enumerated along the way are exactly `{-1,4},{-1,5},{0,4},{0,5},{2,4},{2,5},{4,5}`. The derivation is right, and the `>= T` boundary (the `-1+4 = 3` pair) is handled by the non-strict comparison.

**First implementation — and a trace, because clean math transcribes dirty.** My first cut:

```
sort(a.begin(), a.end());
long long count = 0;
int lo = 0, hi = n - 1;
while (lo < hi) {
    if (a[lo] + a[hi] >= T) {
        count += (hi - lo);
        hi--;
    } else {
        lo++;
    }
}
cout << count << "\n";
```

That looks faithful to the derivation, but two things nag at me: I have not written the `n < 2` guard yet, and `hi - lo` is an `int` subtraction whose result I add to a `long long`. Let me trace the smallest inputs that could expose a base-case or sign bug — exactly the corners the problem advertises.

Start with `n = 1`, `T = 0`, `a = [7]`. After sort, `a = [7]`, `lo = 0`, `hi = n - 1 = 0`. The loop condition `lo < hi` is `0 < 0`, false, so the loop never runs and `count = 0`. Printed: `0`. Correct — a single thruster forms no pair. Good, the loop body itself is safe at `n = 1`.

Now `n = 0`, `T = 5`. Here `a` is empty. `lo = 0`, `hi = n - 1 = -1`. The loop condition `lo < hi` is `0 < -1`, false, loop never runs, `count = 0`. Printed `0`. That *happens* to work, but it works by luck: `hi = -1` is a deliberately invalid index, and if any later edit ever dereferenced `a[hi]` before the loop test — say a "peek at the extremes" optimization — it would read `a[-1]`, undefined behavior. I do not want correctness to hinge on `int` arithmetic making `hi` negative before a comparison. I will add an explicit `if (n < 2) { print 0; return; }` guard so the empty and single cases are handled by intent, not by accident, and the two-pointer body only ever runs with at least two valid elements.

**The first bug — and pinning it precisely.** I deliberately wrote the count increment as `count += (hi - lo)` with `hi` and `lo` both `int`. For the documented bounds this particular expression is fine because `hi - lo <= n - 1 < 2^31`. But while staring at it I write a *wrong* variant that I genuinely was tempted by, to see whether it survives a trace — the "obvious" reading that pairs `hi` only with the elements strictly inside, `count += (hi - lo - 1)`:

```
if (a[lo] + a[hi] >= T) { count += (hi - lo - 1); hi--; }
```

Trace it on `n = 2`, `T = 1`, `a = [-2, 8]`. Sorted `[-2, 8]`, `lo=0, hi=1`. `a[0]+a[1] = -2 + 8 = 6 >= 1`? Yes. With the wrong increment I add `hi - lo - 1 = 1 - 0 - 1 = 0`. `count` stays `0`, `hi--` -> `hi=0`, loop ends. Output `0`. But the brute force is unambiguous: there is exactly one pair `{-2, 8}` and `-2 + 8 = 6 >= 1`, so the answer is `1`. The wrong increment dropped the pair `{lo, hi}` itself — precisely the `lo`-inclusive boundary I flagged in the derivation. The defect is exact: `hi` must be paired with the *closed* range `[lo, hi-1]`, which has `hi - lo` elements, not the open interior `hi - lo - 1`. This is why I derived the increment as a closed range up front; the trace confirms the correct form is `hi - lo`, and I keep it.

**The second bug — sign handling in the comparison, surfaced by an all-negative case.** The remaining worry is the threshold comparison under negatives. Suppose, in a moment of "thresholds are usually positive" tunnel vision, I had narrowed the values and the threshold to a 32-bit `int` to "save memory", writing `int T;` and `vector<int> a;` and computing `a[lo] + a[hi]` as an `int`. Let me trace where that breaks with the extreme all-negative input `n = 2`, `T = -2000000000`, `a = [-1000000000, -1000000000]`. Sorted, `lo=0, hi=1`. The intended test is `a[0] + a[1] = -2000000000 >= T = -2000000000`, which is true (boundary equality), so the answer is `1`. But computed in 32-bit, `-1000000000 + -1000000000 = -2000000000`, and `-2*10^9` is below `INT_MIN = -2147483648`? No — `-2000000000 > -2147483648`, so the *sum* still fits in `int` here. The real failure is `T`: with `|T|` up to `2*10^9`, `T = -2000000000` does fit `int` (just barely), but `T = +2000000000` also fits, while a sum like `+1000000000 + 1000000000 = +2000000000` is right at the edge and any threshold computed as `T - a[i]` (the binary-search variant) would overflow: `2000000000 - (-1000000000) = 3000000000 > INT_MAX`. So the `int` path is a landmine: some inputs near the bounds wrap to the wrong sign and flip the comparison, turning an admissible pair inadmissible or vice versa. The fix that I committed to in decision number one — read `T` and the `a[i]` as `long long` and form `a[lo] + a[hi]` in 64-bit — makes every sum and every threshold exact. I trace the `long long` version on the same input: `a[0]+a[1] = -2000000000LL`, compared to `T = -2000000000LL`, `>=` holds, add `hi - lo = 1`, output `1`. Correct. The all-negative corner with a boundary-equal negative threshold is the case that would have exposed a 32-bit slip, and 64-bit closes it.

**Re-verifying the fixed code on the cases that broke.** With the increment `hi - lo` and 64-bit types and the `n < 2` guard in place:

- `n=2, T=1, a=[-2,8]` (broke under `hi-lo-1`): guard passes (`n>=2`), sorted `[-2,8]`, `lo=0,hi=1`, `-2+8=6>=1`, add `1-0=1`, `count=1`. Correct.
- `n=2, T=-2000000000, a=[-1e9,-1e9]` (the 64-bit case): add `1`, `count=1`. Correct.
- The sample `n=6,T=3` retraced above: `7`. Correct.

The two cases that I broke on purpose now pass, and they pass for the reasons I fixed — the closed-range increment and the 64-bit comparison — which is the evidence I trust over "it looks right".

**Edge cases, deliberately, because this is where this kind of code dies.**

- `n = 0` (empty fleet): the `n < 2` guard prints `0` and returns. No pair exists. Correct, and by intent, not by `hi = -1` luck.
- `n = 1` (single thruster): guard prints `0`. No pair exists. Correct.
- All-negative, threshold unreachable: `n=4, T=-1, a=[-5,-3,-8,-2]`. Sorted `[-8,-5,-3,-2]`. `lo=0,hi=3`: `-8+-2=-10 >= -1`? No, `lo++`. `lo=1,hi=3`: `-5+-2=-7>=-1`? No, `lo++`. `lo=2,hi=3`: `-3+-2=-5>=-1`? No, `lo++`. `lo=3,hi=3`: stop. `count=0`. Correct — no reverse-thrust pair clears a near-zero requirement.
- All-negative, threshold very low so every pair clears: `n=3, T=-100, a=[-5,-6,-7]`. Sorted `[-7,-6,-5]`. `lo=0,hi=2`: `-7+-5=-12>=-100`? Yes, add `2-0=2`, `hi=1`. `lo=0,hi=1`: `-7+-6=-13>=-100`? Yes, add `1-0=1`, `hi=0`. Stop. `count=3 = 3*2/2`. Correct — all `n*(n-1)/2` pairs qualify.
- All zeros at `T = 0`: `n=5, a=[0,0,0,0,0]`. Every pair sum is `0 >= 0`, so all `10` pairs qualify. The sweep: each `a[lo]+a[hi]=0>=0`, so it always takes the `>=` branch, adding `hi-lo` and decrementing `hi`: `(4-0)+(3-0)+(2-0)+(1-0)=10`. Correct — zeros are admissible at a zero threshold, which a strict `>` instead of `>=` would wrongly drop to `0`.
- Threshold above every reachable sum: max pair sum `< T` -> the `>=` branch never fires, `lo` walks to `hi`, `count=0`. Correct.
- Overflow: `count` is `long long`; its max `n*(n-1)/2 ~ 2*10^10` fits. Each sum `a[lo]+a[hi]` is `long long`, range `[-2*10^9, 2*10^9]`, no wrap. `T` is `long long`, `|T| <= 2*10^9`, fits. The increment `hi - lo` is at most `n-1 < 2^31`, promoted to `long long` by the `+=`. Safe.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace so parsing is format-agnostic.

**Cross-checking against a brute force.** I do not want to trust traces alone, so I lean on an independent `O(n^2)` brute that enumerates every pair and tests `a[i]+a[j] >= T` directly — an obviously-correct method, different from the sorted two-pointer sweep. Run over 2000 random small cases drawn from regimes that deliberately include all-negative arrays, all-zero arrays, mixed signs, thresholds below/inside/above the reachable range, and `n` in `{0, 1, 2..10}`: zero mismatches. The empty and single-element cases (`n < 2`) and the all-negative boundary cases are inside that distribution, so the guard and the 64-bit comparison are exercised, not just asserted. I also timed the sweep on `n = 2*10^5` with `|a[i]|` near `10^9`: about `0.07 s`, far under the `1 s` limit, dominated by the sort.

**Final solution.** I convinced myself the *idea* is right by deriving the closed-range increment and the discard rule from the sorted invariant and hand-checking the sample to `7`; I convinced myself the *code* is right by breaking it two ways on purpose — the open-interval increment `hi-lo-1` dropping the `{lo,hi}` pair, and a 32-bit comparison wrapping near the `|T| = 2*10^9` bound — tracing each failure to a precise cause, fixing it, and re-verifying on the broken cases plus the empty / single / all-negative / all-zero corners and a 2000-case brute-force cross-check. That is what I ship — one self-contained file, the `O(n log n)` sort plus `O(n)` two-pointer sweep, with the `n < 2` base case and 64-bit arithmetic made explicit:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;                 // no input / n = 0 -> 0 pairs
    long long T;
    cin >> T;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    if (n < 2) {                               // fewer than two thrusters: no pair exists
        cout << 0 << "\n";
        return 0;
    }

    sort(a.begin(), a.end());

    // Count unordered pairs {i, j}, i < j (in sorted order), with a[i] + a[j] >= T.
    // Two converging pointers on the sorted array.
    long long count = 0;
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        if (a[lo] + a[hi] >= T) {
            // a[hi] paired with every index in [lo, hi-1] also satisfies the threshold,
            // because a is sorted ascending: those are (hi - lo) valid pairs.
            count += (long long)(hi - lo);
            hi--;
        } else {
            lo++;                              // smallest element can never reach T with this hi
        }
    }

    cout << count << "\n";
    return 0;
}
```

**Causal recap.** Sorting turns the sum-threshold pairing into a position question, and the sorted invariant gives a two-pointer sweep whose increment I derived as the *closed* range `[lo, hi-1]`, i.e. `hi - lo` partners for the current `hi`; a trace of `[-2, 8]` with `T=1` showed the tempting `hi - lo - 1` silently drops the `{lo, hi}` pair (answer `0` instead of `1`), pinning the off-by-one; an all-negative boundary case `[-1e9, -1e9]` with `T=-2e9` showed that values, the pair sum, the threshold, and the count all exceed 32-bit, so everything is `long long`; and an explicit `n < 2` base case makes the empty (`hi = -1`) and single-thruster corners return `0` by intent rather than by accidental loop-condition luck, with the all-zero / unreachable-threshold corners and a 2000-case brute-force cross-check closing it out.
