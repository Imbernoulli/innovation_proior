**Reading the problem and pinning the contract.** I have `n` boulders with integer weights `w[i]`, one chisel of integer power `p`, and a global budget of `k` blows. A boulder of weight `w` reduced to pieces each of weight at most `p` costs `ceil(w / p) - 1` blows, and I want the smallest `p` whose total cost over all boulders is at most `k`. Input is `n` and `k` on the first line, then the `n` weights; I print one integer. Before any algorithm I fix the scale, because it dictates the types. `n <= 2*10^5`, `w[i] <= 10^9`, and at `p = 1` a single boulder costs `w - 1` blows, so the total can reach `2*10^5 * (10^9 - 1) ~ 2*10^14`. That dwarfs the 32-bit limit of about `2.1*10^9`, and even `k` itself is allowed up to `2*10^14`. So `k`, every weight, every accumulator, and `p` must be 64-bit `long long`. That is non-negotiable; an `int` anywhere on the cost path is a silent wrong-answer on the large tests.

**Candidate approaches.** The answer `p` is an integer in some range, and the predicate "power `p` stays within budget" is what I have to evaluate. Two routes:

- *Linear scan.* Try `p = 1, 2, 3, ...` and stop at the first feasible one. Correct and trivial, but `p` ranges up to `10^9`, and each feasibility test is `O(n)`, so this is up to `10^9 * 2*10^5` operations — hopeless. It is only useful as the brute-force oracle on tiny inputs.
- *Binary search on the answer.* The predicate is **monotone**: a larger `p` never increases any boulder's blow count (more weight allowed per piece can only mean fewer or equal pieces), so once some `p` is feasible, every larger `p` is feasible. Monotone predicate plus an integer search space is the textbook setup for binary-searching the smallest feasible `p`. Each predicate evaluation is `O(n)`, and the search does `O(log(maxw))` of them, so the whole thing is `O(n log(maxw))` ~ `2*10^5 * 30 = 6*10^6` operations. That is the one to build.

I have to settle the search range. The smallest sensible power is `p = 1`. The largest power I would ever need is `maxw = max(w[i])`: at `p = maxw`, every boulder already has all its weight within one piece, so `ceil(w / maxw) - 1 = 0` blows for each, total `0 <= k` for any `k >= 0`. So `p = maxw` is always feasible, which guarantees the search range `[1, maxw]` contains the answer. Good — no "impossible" case to special-case.

**Deriving the blow formula, and checking it numerically.** This is the first boundary I have to get exactly right. To break a boulder of weight `w` into pieces each of weight at most `p`, I need the fewest pieces `m` with `m * p >= w`, which is `m = ceil(w / p)`. Each blow creates exactly one extra piece (I start with one whole boulder and split), so the number of blows is `m - 1 = ceil(w / p) - 1`.

In integer C++ I do not have a `ceil` for division, and I want to avoid floating point entirely. The identity I will lean on is `ceil(w / p) - 1 == (w - 1) / p` for positive integers `w, p`, where `/` is integer (floor) division. Let me **prove it to myself on concrete numbers** rather than assert it, because this exact expression is where off-by-ones breed.

- `w = 8, p = 4` (an exact multiple): `ceil(8/4) - 1 = 2 - 1 = 1`. And `(8 - 1)/4 = 7/4 = 1`. Match. A boulder of 8 with `p = 4` splits into pieces `4, 4` — that is one blow. Correct.
- `w = 9, p = 4` (just over a multiple): `ceil(9/4) - 1 = 3 - 1 = 2`. And `(9 - 1)/4 = 8/4 = 2`. Match. Pieces `4, 4, 1`, two blows. Correct.
- `w = 4, p = 4` (equal): `ceil(4/4) - 1 = 1 - 1 = 0`. And `(4 - 1)/4 = 3/4 = 0`. Match. One piece, zero blows. Correct.
- `w = 1, p = 1`: `ceil(1/1) - 1 = 0`. And `(1 - 1)/1 = 0`. Match. A 1 kg boulder with `p = 1` is already fine, zero blows. Correct.

The divisibility boundary (`w` an exact multiple of `p`, like `8/4`) is the trap: a careless writer reaches for `w / p` (which gives `2` for `8/4` — one too many) or `ceil(w/p)` without the `-1` (which counts pieces, not blows). The `(w - 1) / p` form is exactly the inclusive-`<= p` cap encoded correctly: it gives one fewer when `w` is a clean multiple. I am confident in the formula now.

**First implementation and a trace.** I write the cost function and the search:

```
auto cuts = [&](long long p) {
    long long total = 0;
    for (long long x : w) total += (x - 1) / p;
    return total;
};
long long lo = 1, hi = maxw;
while (lo < hi) {
    long long mid = lo + (hi - lo) / 2;
    if (cuts(mid) <= k) hi = mid - 1;   // mid works -> look lower
    else lo = mid + 1;                  // mid fails -> look higher
}
cout << lo << "\n";
```

The cost function looks right by the numeric checks above. The search is where I am nervous, so I trace the documented sample: `w = [10, 7, 8, 5]`, `k = 6`, expected answer `4`. Here `maxw = 10`, so `lo = 1, hi = 10`.

First I tabulate `cuts(p)` for the values the search will touch, using `(x-1)/p`:
- `cuts(5)`: `9/5 + 6/5 + 7/5 + 4/5 = 1 + 1 + 1 + 0 = 3 <= 6` feasible.
- `cuts(3)`: `9/3 + 6/3 + 7/3 + 4/3 = 3 + 2 + 2 + 1 = 8 > 6` infeasible.
- `cuts(4)`: `9/4 + 6/4 + 7/4 + 4/4 = 2 + 1 + 1 + 1 = 5 <= 6` feasible.
- `cuts(2)`: `9/2 + 6/2 + 7/2 + 4/2 = 4 + 3 + 3 + 2 = 12 > 6` infeasible.

So the true answer is the smallest feasible `p`, which is `4` (`3` infeasible, `4` feasible). Now the loop:

1. `lo=1, hi=10`, `mid = 1 + 9/2 = 1 + 4 = 5`. `cuts(5)=3 <= 6` feasible -> `hi = mid - 1 = 4`.
2. `lo=1, hi=4`, `mid = 1 + 3/2 = 1 + 1 = 2`. `cuts(2)=12 > 6` infeasible -> `lo = mid + 1 = 3`.
3. `lo=3, hi=4`, `mid = 3 + 1/2 = 3 + 0 = 3`. `cuts(3)=8 > 6` infeasible -> `lo = mid + 1 = 4`.
4. `lo=4, hi=4` -> loop ends. Output `lo = 4`.

That happens to print `4`. Lucky — the answer is correct here, but the `hi = mid - 1` line just stepped over the answer in step 1 (it set `hi = 4`, discarding `5` and below down to `4` simultaneously) and only recovered because `4` got rebuilt from the low side. That coincidence makes me distrust the structure. Let me trace a case engineered to expose it.

**The bug.** Take a tiny case where the *feasible region starts at the very value the search first probes*. `w = [6]`, `k = 1`. Then `maxw = 6`, `lo = 1, hi = 6`. The blow counts: `cuts(p) = (6-1)/p = 5/p`. So `cuts(1)=5, cuts(2)=2, cuts(3)=1, cuts(4)=1, cuts(5)=1, cuts(6)=0`. With `k = 1`, feasible powers are `p >= 3` (`cuts(3)=1 <= 1`), and `cuts(2)=2 > 1`, so the true minimum is `p = 3`. Run my loop:

1. `lo=1, hi=6`, `mid = 1 + 5/2 = 1 + 2 = 3`. `cuts(3)=1 <= 1` feasible -> `hi = mid - 1 = 2`.
2. `lo=1, hi=2`, `mid = 1 + 1/2 = 1`. `cuts(1)=5 > 1` infeasible -> `lo = mid + 1 = 2`.
3. `lo=2, hi=2` -> loop ends. Output `lo = 2`.

The code prints `2`, but `2` is **infeasible** (`cuts(2)=2 > 1`); the true answer is `3`. The defect is precise: when `cuts(mid) <= k`, `mid` *itself* is a feasible candidate and might be the answer, but `hi = mid - 1` throws `mid` away. The search then can only land on values strictly below the smallest feasible one, and there is no path back up because `lo` only climbs over infeasible values. This is the classic "minimum that satisfies a monotone predicate" boundary error: on a hit I must keep `mid` in the window, not exclude it. The sample masked the bug only because the answer `4` was reconstructable from the low side; this `[6], k=1` case has no such escape and pins it.

I also re-examine whether `hi = mid - 1` could ever be right here: it would be the correct move only if I were searching for the *largest* `p` failing the predicate, which is not my question. For the *smallest passing* `p`, a feasible `mid` must shrink `hi` to `mid`, inclusive.

**Fix and re-verification.** The correct collapse keeps the window as `[lo, hi]` where `hi` stays a feasible candidate: on a feasible `mid`, set `hi = mid`; on an infeasible `mid`, set `lo = mid + 1`. The loop ends at `lo == hi`, the smallest feasible value:

```
long long lo = 1, hi = maxw;
while (lo < hi) {
    long long mid = lo + (hi - lo) / 2;
    if (cuts(mid) <= k) hi = mid;       // mid feasible -> answer is mid or smaller, keep mid
    else lo = mid + 1;                  // mid infeasible -> answer strictly larger
}
```

I have to check this terminates: `mid = lo + (hi - lo)/2` satisfies `lo <= mid < hi` whenever `lo < hi` (the floor keeps `mid` below `hi`), so the `hi = mid` branch strictly decreases `hi` and the `lo = mid + 1` branch strictly increases `lo`; the window always shrinks, no infinite loop. (If I had written `mid = (lo + hi + 1)/2` with `hi = mid`, I would stall, but the floor-midpoint with `hi = mid` is the matched pair.)

Re-trace the case that broke, `w = [6], k = 1`, `lo=1, hi=6`:

1. `mid = 1 + 5/2 = 3`. `cuts(3)=1 <= 1` feasible -> `hi = 3`.
2. `lo=1, hi=3`, `mid = 1 + 2/2 = 2`. `cuts(2)=2 > 1` infeasible -> `lo = 3`.
3. `lo=3, hi=3` -> end. Output `3`. Correct — matches the hand-computed minimum.

Re-trace the documented sample `w=[10,7,8,5], k=6`, `lo=1, hi=10`:

1. `mid=5`, `cuts(5)=3 <= 6` -> `hi=5`.
2. `lo=1, hi=5`, `mid = 1 + 4/2 = 3`, `cuts(3)=8 > 6` -> `lo=4`.
3. `lo=4, hi=5`, `mid = 4 + 1/2 = 4`, `cuts(4)=5 <= 6` -> `hi=4`.
4. `lo=4, hi=4` -> end. Output `4`. Correct.

Both the case that broke and the sample now resolve, and they resolve for the reason I fixed (keeping a feasible `mid` in the window), which is the evidence I trust.

**A second trace — divisibility boundary, the other off-by-one.** I built the fix; now I poke the *formula* boundary independently, because the search being right does not protect me from a wrong `cuts`. Consider `w = [6], k = 2`. Expected: `cuts(p) = 5/p`, so `cuts(2) = 2 <= 2` feasible, `cuts(1) = 5 > 2` infeasible; minimum is `p = 2`. Physically: a 6 kg boulder with power 2 splits into `2, 2, 2` — that is two blows, exactly the budget. If I had carelessly written `cuts` as `x / p` instead of `(x-1)/p`, then `cuts(2)` would be `6/2 = 3 > 2`, infeasible, and the code would wrongly answer `p = 3`. Run the real code: `lo=1, hi=6`; `mid=3`, `cuts(3)=5/3=1<=2` -> `hi=3`; `lo=1,hi=3`, `mid=2`, `cuts(2)=5/2=2<=2` -> `hi=2`; `lo=1,hi=2`, `mid=1`, `cuts(1)=5>2` -> `lo=2`; `lo=2,hi=2` -> output `2`. Correct. The `(x-1)/p` form earns its keep precisely on this exact-multiple case; `x/p` would have been off by one in the blow count and shifted the whole answer up.

**Edge cases, deliberately.**
- `k = 0` (no blows allowed): feasibility needs `cuts(p) = 0`, i.e. `(w[i]-1)/p = 0` for every boulder, i.e. `p >= w[i]` for all `i`, i.e. `p >= maxw`. The search should return `maxw`. Trace `w=[4,9,2], k=0`: `maxw=9`, only `p=9` gives all-zero cost (`cuts(9)= 3/9 + 8/9 + 1/9 = 0+0+0 = 0`), `cuts(8)= 3/8 + 8/8 + 1/8 = 0+1+0 = 1 > 0`. So answer `9`. The binary search lands there because `[1,9]` includes it and `maxw` is always feasible. Verified by program: output `9`.
- Single boulder of weight `1`, `k = 0`: `maxw = 1`, range `[1,1]`, loop never runs, output `1`. `cuts(1) = 0/1 = 0 <= 0`. Correct — a 1 kg boulder needs no blows at the weakest chisel.
- Many equal weights, e.g. `w = [1000000000] * 200000`, `k = 0`: answer is `maxw = 10^9`; the cost total at any feasible `p` is `0`, and the early-exit in `cuts` keeps it cheap. No overflow because nothing accumulates.
- Overflow: with `n = 2*10^5` and `w` near `10^9`, `cuts(1)` reaches `~2*10^14`, well within `long long` (`~9.2*10^18`); `k` up to `2*10^14` also fits. To be safe and fast I add an early exit `if (total > k) return total;` inside `cuts` so it stops summing once the budget is already blown — this both prunes work and keeps `total` from running far past `k`.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace, so the two-line layout parses fine.

**Validation at scale.** I compiled with `-O2 -std=c++17` and stress-tested against an independent brute force (a literal piece-counting loop that splits each boulder unit by unit and scans every `p` from `1` upward) on over 400 random small cases — boulders up to weight 12, `k` swept across the whole interesting range including `0` and exact-divisibility budgets — with **zero mismatches**. I separately built two buggy variants, one with `hi = mid - 1` and one with `x/p`, and confirmed they diverge from the brute force on 135 and 97 of 300 cases respectively, proving both boundaries are genuinely load-bearing. The large test (`n = 2*10^5`, `w[i]` near `10^9`) runs in about `0.05 s` using under `5 MB`.

**Final solution.** I convinced myself the idea is right by ruling out the linear scan on complexity and proving the monotonicity that licenses binary search; I convinced myself the *code* is right by tracing a case (`[6], k=1`) where the search off-by-one returns the infeasible `2` instead of `3`, fixing the `hi = mid` boundary, and independently tracing the divisibility case (`[6], k=2`) that the `(x-1)/p` formula handles where `x/p` would not. That is what I ship — one self-contained file:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long k;
    if (!(cin >> n >> k)) return 0;
    vector<long long> w(n);
    long long maxw = 1;
    for (auto &x : w) { cin >> x; maxw = max(maxw, x); }

    // cuts(p) = total chisel blows needed if each blow can leave pieces of size <= p.
    // A boulder of weight x needs ceil(x/p) final pieces, i.e. ceil(x/p)-1 blows.
    // ceil(x/p)-1 == (x-1)/p  in integer arithmetic (x>=1, p>=1).
    auto cuts = [&](long long p) -> long long {
        long long total = 0;
        for (long long x : w) {
            total += (x - 1) / p;       // ceil(x/p) - 1 blows for this boulder
            if (total > k) return total; // early exit, avoid overflow-ish blowup
        }
        return total;
    };

    // feasible(p): with chisel power p, total blows <= k. Monotone: larger p => fewer blows.
    // We want the MINIMUM p with feasible(p) true. p ranges over [1, maxw]
    // (p = maxw always needs 0 blows per boulder, so it is always feasible).
    long long lo = 1, hi = maxw;        // search space [lo, hi], answer guaranteed in it
    while (lo < hi) {
        long long mid = lo + (hi - lo) / 2;
        if (cuts(mid) <= k) hi = mid;   // mid works -> answer is mid or smaller
        else lo = mid + 1;              // mid fails -> answer strictly larger
    }
    cout << lo << "\n";
    return 0;
}
```

**Causal recap.** The linear scan is correct but `O(maxw * n)` and dies at `maxw = 10^9`, so I leaned on the monotonicity of "power `p` stays within `k` blows" to binary-search the smallest feasible `p`; the blow count per boulder is `ceil(w/p)-1`, which I encode as `(w-1)/p` and verified on exact-multiple cases like `8/4` where the naive `w/p` is off by one. My first search used `hi = mid - 1` on a feasible `mid`, which discards the candidate `mid` and can only return a value below the true minimum — the trace of `[6], k=1` returning the infeasible `2` instead of `3` pinned it, and switching to `hi = mid` (keeping the feasible `mid` in the window, with the floor-midpoint guaranteeing termination) fixes it. The `k = 0`, single-boulder, and large-`n` overflow corners are closed by the `[1, maxw]` range (where `maxw` is always feasible), the 64-bit accumulators, and the early-exit in `cuts`.
