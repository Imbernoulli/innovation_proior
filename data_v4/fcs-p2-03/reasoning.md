**Reading the problem and pinning the contract.** I am given `n` integers `a[0..n-1]` that may be negative, zero, or positive, and I have to look over every contiguous non-empty subarray `a[i..j]`, compute its product, and report the largest product I can find. The empty subarray is *not* allowed, which matters: it means a single element is always a legal window, so on a pathological input the answer can just be the largest single value. Input is `n` then the `n` values on stdin; I print one integer. Before reaching for any algorithm I fix the scale, because that decides the data types. The constraints say `1 <= n <= 18` and `-9 <= a[i] <= 9`. The biggest a product can get in magnitude is all eighteen elements at magnitude nine, i.e. `9^18`. Let me actually compute that rather than eyeball it: `9^18 = 150094635296999121`, about `1.5 * 10^17`. A signed 64-bit integer tops out near `9.2 * 10^18`, so every reachable product — positive or negative — fits in `long long` with more than an order of magnitude of headroom. That is the first non-negotiable decision: 64-bit accumulators throughout. A 32-bit `int` overflows the moment a window's product passes `~2.1 * 10^9`, which happens quickly at these magnitudes, and that is a silent wrong answer on exactly the large tests the judge will throw. So `long long` it is, and I never need big integers or floating point.

**Laying out the candidate approaches.** Two routes are on the table, and I want to commit to the one I can *prove*, not the one that is fastest to type.

- *Kadane-on-product.* The maximum-*sum* subarray problem has a famous one-line recurrence: carry `cur = max(a[i], cur + a[i])` and track the running best. The product problem looks like the same shape with `+` replaced by `*`: `cur = max(a[i], cur * a[i])`, track the global max. It is `O(n)`, three lines, and feels like the "clever" textbook move. The risk is structural and specific to multiplication: with addition, a larger running value is always at least as good to extend; with multiplication, multiplying by a *negative* number reverses order, so the value you most want to extend through a negative is the *smallest* (most negative) running product, not the largest. Carrying only `cur = max(...)` throws that information away. I do not trust this until I have tried to break it.
- *Min/max product DP.* Scan left to right and carry, for the subarray *ending at* the current index, both the maximum product `curMax` and the minimum product `curMin`. When I see a new element `x`, the best window ending at `x` is one of three things: start fresh at `x`; extend the previous best, `curMax * x`; or extend the previous worst, `curMin * x` — and that third candidate is exactly the one that rescues a negative `x` by flipping a large negative running product into a large positive. `O(n)`, `O(1)` memory. The risk here is not the soundness of the idea but getting the candidate set and the base case transcribed correctly.

**Stress-testing Kadane-on-product before committing.** "It's just Kadane with times instead of plus" is precisely the kind of analogy that ships wrong solutions, so let me attack it with a concrete instance rather than wave at it. Take the all-negative array `a = [-1, -2, -3, -4, -5]`. The true best is the product of all five — wait, five negatives multiply to a negative, `-120`. The best *positive* window is the largest even-length run: `[-2, -3, -4, -5]` gives `120`, and so does `[-1,-2,-3,-4]` which is `24`... let me just compute: `(-2)*(-3)*(-4)*(-5) = 120`. So the answer is `120`. Now run Kadane-on-product by hand. `cur = -1`, `best = -1`. i=1, x=-2: `cur = max(-2, (-1)*(-2)) = max(-2, 2) = 2`, `best = 2`. i=2, x=-3: `cur = max(-3, 2*(-3)) = max(-3, -6) = -3`, `best = 2`. i=3, x=-4: `cur = max(-4, (-3)*(-4)) = max(-4, 12) = 12`, `best = 12`. i=4, x=-5: `cur = max(-5, 12*(-5)) = max(-5, -60) = -5`, `best = 12`. Kadane-on-product returns `12`, the true answer is `120`. It is wrong, by a factor of ten.

And I can see *exactly why*. At i=2 the running window product became `-3` (just the element `-3` itself, because extending the stored `+2` through `-3` gave the worse `-6`). But the genuinely valuable quantity at that step was the *minimum* product ending there, which was `2 * (-3) = -6` — a large-magnitude negative. One step later, multiplying that `-6` by `-4` would have produced `24`, and from there `* -5` would have produced `-120`... no, the chain that actually wins is `(-2)(-3)(-4)(-5)`. The point stands regardless of which exact window: by keeping only the running maximum, Kadane-on-product discards the most-negative running product, and the most-negative running product is precisely what a future negative factor turns into the new maximum. The verification paid off — it killed an approach I would otherwise have shipped as "obviously Kadane."

Let me confirm with a second, tiny witness so I am not fooling myself with one cherry-picked array: `a = [-2, 3, -4]`. True best is the whole array, `(-2)*3*(-4) = 24`. Kadane-on-product: `cur=-2, best=-2`. i=1,x=3: `cur=max(3, -6)=3`, `best=3`. i=2,x=-4: `cur=max(-4, 3*(-4))=max(-4,-12)=-4`, `best=3`. It returns `3`; truth is `24`. Same failure mode: at i=1 the min product ending there was `-2 * 3 = -6`, discarded, and `-6 * -4 = 24` is the answer Kadane never sees. Kadane-on-product is out.

**Deriving the min/max DP and checking the recurrence on paper.** I want, for each ending index `i`, the maximum product of a window that ends at `i`; the global answer is the max of that over all `i`. The only facts about the prefix that the next step cares about are the best and worst products of a window ending at `i`, because a window ending at `i+1` either starts at `i+1` or extends a window ending at `i`. So I carry exactly two numbers:

- `curMax` = maximum product over windows ending at the current index,
- `curMin` = minimum product over windows ending at the current index.

Transition at element `x = a[i]`. A window ending at `i` is either the singleton `{i}` with product `x`, or an extension of a window ending at `i-1` by `x`. Extending the previous-max gives `curMax * x`; extending the previous-min gives `curMin * x`. So the three candidates are `c1 = x`, `c2 = curMax * x`, `c3 = curMin * x`, and:

- `curMax_i = max(c1, c2, c3)`,
- `curMin_i = min(c1, c2, c3)`.

Both new values are computed from the *previous* `(curMax, curMin)` pair. The global answer is the running maximum of `curMax_i` over all `i`. Base case: at `i = 0` the only window ending there is `{0}` itself, so `curMax = curMin = best = a[0]`. This is well-defined because the constraints guarantee `n >= 1`, so `a[0]` always exists — I do not need to invent a neutral product (which for multiplication would be `1`, and a phantom `1` would be wrong since the empty window is disallowed). Starting the scan at index `1` with the index-`0` seed sidesteps that trap entirely.

Let me confirm the recurrence by hand on the statement's sample `a = [2, 3, -2, 4]`, answer `6`. Seed at i=0: `curMax = curMin = best = 2`. i=1, x=3: `c1=3, c2=2*3=6, c3=2*3=6`; `curMax=6, curMin=3, best=6`. i=2, x=-2: `c1=-2, c2=6*(-2)=-12, c3=3*(-2)=-6`; `curMax=max(-2,-12,-6)=-2, curMin=min(...)=-12, best=max(6,-2)=6`. i=3, x=4: `c1=4, c2=(-2)*4=-8, c3=(-12)*4=-48`; `curMax=max(4,-8,-48)=4, curMin=-48, best=max(6,4)=6`. Final `6`. Correct, and notice at i=2 the genuinely useful number for the future was `curMin=-12`; it is carried, even though it lost to `6` for the answer — which is the whole reason this beats Kadane.

And let me re-run the array that broke Kadane, `[-2, 3, -4]`, through this DP to be sure the fix lands. Seed i=0: `curMax=curMin=best=-2`. i=1,x=3: `c1=3, c2=-6, c3=-6`; `curMax=3, curMin=-6, best=3`. i=2,x=-4: `c1=-4, c2=3*(-4)=-12, c3=(-6)*(-4)=24`; `curMax=max(-4,-12,24)=24, curMin=-12, best=max(3,24)=24`. Final `24`. The candidate `c3 = curMin * x = (-6)(-4) = 24` is exactly the negative-times-negative flip Kadane threw away, and the DP captures it because it carried `curMin`.

**First implementation — and a trace, because clean math transcribes dirty.** My first cut of the loop body, writing the two updates in place:

```
long long curMax = a[0], curMin = a[0], best = a[0];
for (int i = 1; i < n; i++) {
    long long x = a[i];
    curMax = max({x, curMax * x, curMin * x});
    curMin = min({x, curMax * x, curMin * x});   // BUG: uses the just-updated curMax
    best = max(best, curMax);
}
```

Something about computing `curMax` and then reusing `curMax` in the `curMin` line on the same iteration looks dangerous — the same in-place hazard that bites every two-variable DP. I trace the smallest input that could expose it: `a = [-3, -4]`, where the answer is obviously `12` (two negatives, whole array). Seed i=0: `curMax=curMin=best=-3`. i=1, x=-4: `curMax = max{-4, (-3)*(-4), (-3)*(-4)} = max{-4, 12, 12} = 12`. Then `curMin = min{-4, curMax*x, curMin*x}` but `curMax` is now `12`, so `curMax*x = 12*(-4) = -48`, and `curMin*x = (-3)*(-4) = 12`, giving `curMin = min{-4, -48, 12} = -48`. Final `best = 12`. On this particular array the *answer* `12` came out right, but `curMin` is now `-48`, which is garbage: there is no window ending at index 1 with product `-48` (the windows are `[-4]=-4` and `[-3,-4]=12`; the true min is `-4`). The corrupted `curMin` is a landmine for any longer array.

**Diagnosing the bug and constructing the case that detonates it.** To make the corruption actually change the output I need a third element that multiplies the bogus `curMin = -48` into something that beats the real answer, or — more insidiously — a case where the bogus `curMin` is *too negative* and produces a phantom large product. Take `a = [-3, -4, -5]`. True answer: best even-length window. `[-3,-4]=12`, `[-4,-5]=20`, `[-3,-4,-5]=-60`. So the truth is `20`. Trace the buggy code. Seed i=0: `(-3,-3,-3)`. i=1,x=-4: `curMax = max{-4, 12, 12} = 12`; then with corrupted `curMax=12`, `curMin = min{-4, 12*(-4), (-3)*(-4)} = min{-4, -48, 12} = -48`. State `(curMax,curMin)=(12,-48)`, `best=12`. i=2,x=-5: `curMax = max{-5, 12*(-5), (-48)*(-5)} = max{-5, -60, 240} = 240`; `best = 240`. The buggy code outputs `240`. The true answer is `20`. The phantom `240 = (-48)*(-5)` came from a `curMin` of `-48` that corresponds to no real window — it was manufactured by reading the freshly-updated `curMax` instead of the old one. So the bug is not cosmetic; it fabricates impossible products and reports them.

The defect is precise: both transitions must read the *previous* `(curMax, curMin)` pair, but writing `curMax` first and then reading `curMax` in the `curMin` line feeds the new max back into the min computation. I destroyed `curMax` before I was done reading it.

**Fixing and re-verifying.** Snapshot the two old values into temporaries (`c2`, `c3` already capture the only products that use them), compute both new values from the old pair, then assign:

```
long long c1 = x;
long long c2 = curMax * x;     // both read the OLD curMax / curMin
long long c3 = curMin * x;
curMax = max(c1, max(c2, c3));
curMin = min(c1, min(c2, c3));
```

Re-trace the detonating case `[-3, -4, -5]`. Seed i=0: `(-3,-3)`, best=-3. i=1,x=-4: `c1=-4, c2=(-3)(-4)=12, c3=(-3)(-4)=12`; `curMax=12, curMin=min{-4,12,12}=-4`, best=12. (Now `curMin=-4` is the *correct* min for windows ending at index 1, not `-48`.) i=2,x=-5: `c1=-5, c2=12*(-5)=-60, c3=(-4)*(-5)=20`; `curMax=max{-5,-60,20}=20, curMin=min{-5,-60,20}=-60`, best=max(12,20)=20. Final `20`. Correct, and the case that fabricated `240` before now reports the true `20`, for the exact reason I fixed. That is the evidence I trust: it broke for a diagnosed cause and the fix addresses that cause.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 1`, `a = [-5]`: the loop never runs; `best = a[0] = -5`. The only window is the singleton, and the empty window is disallowed, so a lone negative *must* return itself, not `0`. Correct — and this is why I seed from `a[0]` rather than initialize to `0` or `1`.
- `n = 1`, `a = [0]`: `best = 0`. Correct.
- All negative even count, `[-2,-3,-4,-5]`: the whole array is positive (`120`) and the DP's `curMin`-carrying recovers it. (Checked by my differential harness below.)
- A zero in the middle, `[-9, 0, -9, 0, -9]`: a `0` makes `c2`/`c3` zero, and `c1 = x` lets the window restart cleanly after the zero; the best here is `0` (every multi-element window through a zero is `0`, and the best single element is `0` while the negatives are worse). The DP handles the reset because `c1 = x` is always a candidate, so the window can "start fresh" at any index.
- Overflow: accumulators are `long long`; the maximum magnitude `9^18 ≈ 1.5*10^17` fits with room to spare. The products `c2`, `c3` are formed from values already bounded by that magnitude, so no intermediate exceeds 64 bits. Safe.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace, so spacing and line breaks in the input do not matter.

**Self-verification at scale.** Hand traces convince me of the idea; a differential test convinces me of the *code*. I wrote an independent brute oracle in Python — the naive `O(n^2)` double loop over every `(i, j)` with Python's exact big integers, sharing no logic with the `O(n)` DP — plus a generator covering seven distributions (generic mixed signs, sign-only `{-2..2}`, all-positive, all-negative, zero-sprinkled, tiny `n` for the `n=1,2` corners, and maximal `n=18` at `±9`). Compiling the C++ and running 13 hand-built edge cases plus 600 random cases gave **zero mismatches**. I then ran an exhaustive sweep over *every* array with `n <= 4` and values in `[-3, 3]` — 2800 arrays — again **zero mismatches**. The two arrays that broke Kadane-on-product (`[-1,-2,-3,-4,-5]` → 120 vs Kadane's 12; `[-2,3,-4]` → 24 vs Kadane's 3) are produced correctly by the shipped DP. The brute and the DP agree everywhere I can check.

**Final solution.** I convinced myself the idea is right by disproving Kadane-on-product with two concrete counterexamples and hand-checking the min/max recurrence on the sample, and I convinced myself the *code* is right by constructing the array that detonates the in-place update bug, tracing it to a precise cause, fixing it, and then differential-testing the fix to zero mismatches over 600 random + 13 edge + 2800 exhaustive cases. That is what I ship — one self-contained file, the simple `O(n)` min/max DP I can defend rather than the Kadane variant I broke:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // Track, for the subarray ending at the current position, BOTH the maximum
    // and the minimum product. A negative current element swaps their roles
    // (min*neg can become the new max), so we must carry the minimum too.
    long long curMax = a[0];      // best product of a subarray ending here
    long long curMin = a[0];      // worst product of a subarray ending here
    long long best   = a[0];      // global answer
    for (int i = 1; i < n; i++) {
        long long x = a[i];
        long long c1 = x;             // start fresh at i
        long long c2 = curMax * x;    // extend previous best
        long long c3 = curMin * x;    // extend previous worst (key for negatives)
        curMax = max(c1, max(c2, c3));
        curMin = min(c1, min(c2, c3));
        best = max(best, curMax);
    }

    cout << best << "\n";
    return 0;
}
```

**Causal recap.** Kadane-on-product looked like the clever textbook move but two traced counterexamples (`[-1,-2,-3,-4,-5]`: Kadane 12 vs reachable 120; `[-2,3,-4]`: Kadane 3 vs reachable 24) showed that carrying only the running maximum discards the most-negative running product, which is exactly what a future negative factor turns into the new maximum — so I moved to the min/max DP that carries both. Its two transitions both read the *previous* `(curMax, curMin)` pair, which my first in-place loop destroyed by overwriting `curMax` before reading it for `curMin`; an array I constructed for the purpose, `[-3,-4,-5]`, fabricated a phantom `240` against the true `20`, pinpointing the hazard; computing both new values from the old pair via `c1,c2,c3` and seeding from `a[0]` (legal since `n>=1`, and the empty window is disallowed) fixes it; and `long long` accumulators over the `9^18 ≈ 1.5*10^17` magnitude bound close out overflow, with 600 random + 13 edge + 2800 exhaustive differential checks against an independent brute oracle confirming zero mismatches.
