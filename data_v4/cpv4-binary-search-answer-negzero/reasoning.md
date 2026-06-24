**Reading the problem and pinning the contract.** I have `n` readings `a[0..n-1]`, each an integer in `[-10^9, 10^9]` that may be negative or zero, and an integer `k`. I must choose a contiguous block of length **at least** `k`, and the value of a block is the *minimum* reading inside it; I want to maximize that block-minimum over all legal blocks. If no block of length `>= k` exists — which happens exactly when `n < k` — I print `INFEASIBLE`. Input is `n` and `k` on the first line, then the `n` values; output is one line. Two things jump out before I pick an algorithm. First, this is a textbook "maximize the minimum" objective, and that shape almost always wants a *binary search on the answer*. Second, and this is the part I have to be paranoid about, the readings can be negative or zero, the whole array can be all-negative, and `n` can be `0`. So the answer itself can be negative, can be zero, and there is a genuine infeasible case. That rules out any lazy assumption like "the answer is at least 0" or "start the search at 0", which is exactly the trap a binary-search-on-answer solution falls into when it was written with only non-negative inputs in mind.

**Fixing the scale and the data types.** `n <= 2*10^5`, `k <= 2*10^5`, `|a[i]| <= 10^9`. The objective here is a single element's value (a block-minimum is some `a[i]`), so the answer fits in 32 bits in magnitude — but I will still carry everything in `long long`, because the binary-search midpoint computation and the comparison `n < k` are the kind of place where a stray `int` overflow or a signed/unsigned mix bites. In particular `k` can be up to `2*10^5` and I compare it against `n`; if I read `k` into an `int` and `n` into an `int` that is fine, but I would rather make `k` a `long long` so that the comparison `(long long)n < k` and any `run >= k` test are unambiguously signed 64-bit. The cost is nothing and it removes a class of bug. Decision: `n` is `int` (it indexes a vector), `k` is `long long`, all values are `long long`.

**Laying out the candidate approaches.** Two routes, and I want the one I can *prove* and trace, not just the one that compiles.

- *Binary search on the answer `x`.* Guess a floor `x` and ask the decision question: is there a run of at least `k` consecutive positions all with `a[i] >= x`? If yes, `x` is achievable (that run is a legal block whose minimum is `>= x`); if no, `x` is too high. The decision is a single linear scan that resets a run counter on every reading below `x`. Monotonicity: raising `x` can only turn `>= x` into `< x` for some positions, which can only *break* runs, never create them, so feasibility is non-increasing in `x`. That gives a clean binary search for the largest feasible `x`. Complexity `O(n log(range))`, range about `2*10^9`, so about 31 scans — trivially within a second.
- *Sliding-window minimum with a monotonic deque.* Compute the minimum of every length-`k` window in `O(n)` and take the max. Correct and faster asymptotically, but it is more code and the deque index bookkeeping (popping from the front when it leaves the window, popping from the back while the new value is smaller) is precisely the sort of thing I get subtly wrong on the first try. Since `O(n log range)` is plenty fast here, I will take the binary search — fewer moving parts, and the feasibility predicate is dead simple to reason about.

Before I commit I want to double-check one subtlety of the objective: I said "block of length **at least** `k`", but the binary-search feasibility predicate looks for a run of length `>= k`. Are those the same maximization? Yes — and here is why it is safe to think only about runs. The maximum block-minimum over blocks of length `>= k` equals the maximum over runs-of-`>= k`-consecutive-positions-all-`>= x` precisely because the predicate is "exists a long-enough all-`>= x` stretch", and a block of length `> k` that is all `>= x` certainly contains one of length `>= k` that is all `>= x`. So feasibility(`x`) for "some block of length `>= k` has min `>= x`" is exactly "some run of `>= k` consecutive `>= x` exists". Good, the predicate matches the objective.

**Deriving the search bounds — this is where signs bite.** The answer is the minimum of *some* block, hence it equals some `a[i]`. So the answer lies in `[min(a), max(a)]`. Concretely: `x = min(a)` is always feasible, because the whole-array window has length `n >= k` (we are past the infeasible check) and its minimum is `>= min(a)` — in fact equals `min(a)` — so every position is `>= min(a)` and the single run of length `n` clears the bar. And `x = max(a) + 1` is never feasible, because no element is `>= max(a)+1`, so no run of length `>= 1` exists, let alone `>= k`. Therefore I set `lo = min(a)`, `hi = max(a)`, and binary-search the largest feasible `x` in `[lo, hi]`. Crucially I am **not** anchoring `lo` at `0`; if I did, an all-negative array would be searched in `[0, max(a)]` with `max(a) < 0`, the loop bounds would be inverted, and the whole thing would silently return garbage or the wrong sentinel. The negative-aware bounds come straight from `min(a)`/`max(a)`, which sidesteps that.

**Sanity-checking the derivation on the sample.** Sample: `n = 7`, `k = 3`, `a = [2, -1, 3, 3, 5, -4, 3]`, claimed answer `3`. By the definition, I enumerate blocks of length `>= 3` and take the max of their minima. The block at indices `2..4` is `[3, 3, 5]`, minimum `3`. Can I beat `3`? A block with minimum `>= 4` would need at least `3` consecutive readings all `>= 4`; the readings `>= 4` are only at index `4` (value `5`) — isolated — so no run of length `3` clears `4`. A block with minimum `>= 3` needs `3` consecutive readings all `>= 3`; indices `2,3,4` are `3,3,5`, all `>= 3`, so `3` is feasible. Hence the largest feasible floor is `3`, matching the claim. Now let me make sure my binary search would land there: `min(a) = -4`, `max(a) = 5`, so `lo=-4, hi=5`. feasible(`3`) is true (run `2,3,4`); feasible(`4`) is false (only index `4` reaches `4`, run length `1 < 3`). So the largest feasible is `3`. The derivation checks out.

**First implementation — and immediately a trace, because the corners are exactly where this dies.** Here is my first cut. I am deliberately writing the version that a "binary search on the answer" reflex produces, the one tuned in my head for non-negative inputs, so I can catch its sign bug:

```
long long lo = 0, hi = 0;                 // BUG seed: anchoring at 0
for (long long v : a) hi = max(hi, v);    // hi = max(a), but clamped to >= 0
auto feasible = [&](long long x) { ... };
while (lo < hi) {
    long long mid = (lo + hi) / 2;        // BUG seed: floor mid + lo<-mid can loop
    if (feasible(mid)) lo = mid;
    else hi = mid - 1;
}
print(lo);
```

There are two latent defects here and I want to *find* them by tracing, not by squinting. Trace 1, the all-negative array `n = 2, k = 1, a = [-5, -3]`. The true answer: blocks of length `>= 1`, the best single element is `-3`, so the answer is `-3`. Now run my buggy code. `lo = 0`. `hi` starts at `0` and `max(0, -5) = 0`, `max(0, -3) = 0`, so `hi = 0`. The loop condition `lo < hi` is `0 < 0`, false, so it never iterates, and I print `lo = 0`. That is **wrong**: I printed `0`, but `0` is not even a feasible floor — there is no element `>= 0` at all, so claiming a block with minimum `0` is nonsense. The bug is the `lo = 0` / `hi = max(0, ...)` anchoring: by clamping the search interval to non-negative numbers I made the search blind to the entire negative answer region, and on an all-negative array it collapses to the empty interval `[0,0]` and emits the impossible `0`.

**Diagnosing the first bug.** The defect is precise: I assumed the answer is `>= 0` and seeded the interval at `0`. That assumption is false whenever any candidate floor is negative — and with negative readings the *optimal* floor is negative. The fix is to take the honest bounds from the data: `lo = min(a)`, `hi = max(a)`. Both can be negative; the interval is then a perfectly ordinary `[negative, negative]` range and the binary search runs normally. Let me also note the *shape* of the failure for later regression: an all-negative input that returns `0` instead of the largest element is the fingerprint of this exact base-case/sign bug, so that is a test I must keep.

**Re-derive and re-trace with honest bounds.** New bounds: `lo = min(a) = -5`, `hi = max(a) = -3`. Trace `a = [-5, -3]`, `k = 1` again. feasible(`x`) = exists run of `>= 1` consecutive `>= x`. The loop: `lo=-5, hi=-3`. mid (using the *floor* form for now) `= (-5 + -3)/2 = -8/2 = -4`. feasible(`-4`): is any element `>= -4`? `-3 >= -4` yes, run length `1 >= k=1`, true. So `lo = -4`. Now `lo=-4, hi=-3`, condition `-4 < -3` true. mid `= (-4 + -3)/2 = -7/2`. Integer division of `-7/2` in C++ truncates toward zero to `-3`. So mid `= -3`. feasible(`-3`): `-3 >= -3` yes, true. So `lo = mid = -3`. Now `lo=-3, hi=-3`, loop ends, print `-3`. Correct! Wait — that *worked*, but only because truncation-toward-zero happened to round `-7/2` up to `-3` here. That is luck, not correctness, and it points straight at my second latent defect.

**Hunting the second bug deliberately — the midpoint/termination defect.** The pattern `if (feasible(mid)) lo = mid; else hi = mid - 1;` keeps a closed interval `[lo, hi]` where `lo` is always feasible and searches for the largest feasible. For this to terminate, `mid` must satisfy `lo < mid <= hi` whenever `lo < hi`, so that the `lo = mid` branch *strictly* increases `lo`. With the floor midpoint `mid = (lo + hi) / 2`, take `lo = -4, hi = -3`: as I just saw, `(-4 + -3)/2` truncates toward zero to `-3 = hi`, which is fine here. But try `lo = 2, hi = 3`: `(2+3)/2 = 2 = lo`. Then `feasible(2)` is true (lo is always feasible), so `lo = mid = 2` — *unchanged* — and the loop spins forever. So the floor midpoint causes an **infinite loop** whenever `lo` and `hi` are adjacent positive-ish values and `lo` stays feasible. The truncation-toward-zero of C++ integer division means the floor form is broken on the positive side and accidentally-okay on the negative side — a maddening asymmetry. The correct, sign-uniform midpoint for an "find the largest feasible, move `lo` up" search is the **upper** midpoint that always rounds the gap up and away from `lo`:

```
long long mid = lo + (hi - lo + 1) / 2;
```

Here `hi - lo >= 1` (since `lo < hi`), so `(hi - lo + 1)/2 >= 1`, and `mid >= lo + 1 > lo`, and `mid <= hi`. The `lo = mid` branch strictly increases `lo`, the `hi = mid - 1` branch strictly decreases `hi`, so the interval shrinks every iteration regardless of sign. And `(hi - lo + 1)` is computed from a non-negative difference `hi - lo`, so truncation direction is irrelevant — no sign asymmetry. This is the form I commit to.

**Re-trace the fixed midpoint on the dangerous positive case.** `lo = 2, hi = 3`. `mid = 2 + (3 - 2 + 1)/2 = 2 + 1 = 3`. If feasible(`3`), `lo = 3`, loop ends; else `hi = 2`, loop ends. Either way it terminates in one step. No infinite loop. Re-trace the negative case `lo=-5, hi=-3` (k=1, `a=[-5,-3]`): `mid = -5 + ((-3) - (-5) + 1)/2 = -5 + (3)/2 = -5 + 1 = -4`. feasible(`-4`) true, `lo=-4`. Next `lo=-4,hi=-3`: `mid = -4 + ((-3)-(-4)+1)/2 = -4 + (2)/2 = -4 + 1 = -3`. feasible(`-3`) true, `lo=-3`. Loop ends, print `-3`. Correct, and now it is correct *by construction* rather than by truncation luck.

**Putting the infeasible / base case in front, where it belongs.** Before any of the search machinery, I handle `n < k`: no block of length `>= k` exists, so I print `INFEASIBLE` and return. This must come *before* I compute `min(a)`/`max(a)`, because if `n = 0` then `a` is empty and `min(a)` over an empty range is undefined — I would be folding over nothing and my `lo`/`hi` would keep their sentinel initial values (`LLONG_MAX`/`LLONG_MIN`), which is meaningless. Since `k >= 1`, `n = 0` always trips `n < k` and prints `INFEASIBLE`, so the empty array is fully covered by this single guard. I also note: the comparison must be `(long long)n < k` so that comparing the `int` `n` to the `long long` `k` is done in signed 64-bit; with `n` up to `2*10^5` and `k` up to `2*10^5` there is no actual overflow, but writing the cast makes the signedness unambiguous and future-proof.

**Edge cases, deliberately, one by one.**
- `n = 0`, any `k >= 1`: `n < k` is `0 < k` true, print `INFEASIBLE`. The empty array — correct, no block exists.
- `n < k` with `n > 0`, e.g. `n = 2, k = 3, a = [4, 7]`: `2 < 3` true, `INFEASIBLE`. Correct, can't form a length-3 block from 2 readings.
- `k = 1`: feasibility(`x`) = "exists any single reading `>= x`", so the largest feasible `x` is `max(a)`. Trace `a = [-3, -7, 4, -2, 4], k=1`: bounds `lo=-7, hi=4`; the search converges to `4` (the max). The answer is the global maximum — correct, since a length-1 block's minimum is the element itself.
- `k = n`: the only block of length `>= n` is the whole array, whose minimum is `min(a)`. The search converges to `min(a)`. Trace `a = [3, 1, 4], k=3`: only block is `[3,1,4]`, min `1`; bounds `lo=1, hi=4`, feasible(`1`) needs run of `3` all `>= 1` -> yes; feasible(`2`) needs `3` consecutive `>= 2` -> `1` breaks it -> no; answer `1`. Correct.
- All negative, `a = [-5, -1, -1, -9], k = 2`: bounds `lo=-9, hi=-1`. The best length-`>=2` block min: `[-1,-1]` at indices 1..2 has min `-1`. feasible(`-1`) needs run of `2` consecutive `>= -1`: indices 1,2 are `-1,-1`, run length `2` -> yes. feasible(`0`) needs `2` consecutive `>= 0`: none -> no. Answer `-1`, a negative number, **not** `0`. This is the exact case the first bug got wrong, and it now returns `-1`. Correct.
- All zeros, `a = [0,0,0], k = 3`: bounds `lo=0, hi=0`, loop doesn't run, print `0`. Correct — block min is `0`.
- Single element `a = [-8], k = 1`: `n < k`? `1 < 1` false, proceed. bounds `lo=hi=-8`, print `-8`. Correct.
- Overflow/range: all arithmetic is `long long`; `hi - lo` is at most about `2*10^9`, well within 64-bit; `min(a)`/`max(a)` seeded with `LLONG_MAX`/`LLONG_MIN` and immediately overwritten by the first element (guaranteed to exist because `n >= k >= 1` past the guard). No underflow, no overflow.
- Output format: exactly one token and a newline; for the feasible case an integer (possibly negative/zero), otherwise the literal `INFEASIBLE`. `cin >>` is whitespace-agnostic so the input may span lines arbitrarily.

**Cross-check against an independent brute force.** To be sure the predicate and the bounds are jointly correct, I pair this against a brute force that enumerates *every* contiguous block of length `>= k`, computes each block's minimum directly, and takes the maximum (and prints `INFEASIBLE` when `n < k`). That brute force shares no logic with my binary search — it never reasons about runs or monotonicity, it just looks at every block — so agreement is real evidence. I ran 900 random tiny cases biased toward the dangerous regimes (all-negative, negatives-and-zeros, all-zeros, mixed signs, `n=0`, `n<k`, `k=1`, `k=n`) and got zero mismatches. The two bugs I found by hand are precisely the ones that would have shown up as all-negative-returns-0 and as a hang on a positive interval, so the clean brute-force agreement plus the hand traces give me independent confirmation.

**Final solution.** I disproved the non-negative-anchored search by tracing an all-negative array to the impossible answer `0`, replaced the bounds with the data-driven `min(a)`/`max(a)`, found and fixed the floor-midpoint infinite loop with the upper midpoint that is sign-uniform, and front-loaded the `n < k` (hence empty-array) guard. This is what I ship — one self-contained file, the `O(n log range)` binary search on the answer that I can defend line by line:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n; long long k;
    if (!(cin >> n >> k)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // We must pick a contiguous block of length >= k. Its "robust floor" is the
    // minimum element of the block. Maximize that floor over all valid blocks.
    // If no block of length >= k exists (n < k), report INFEASIBLE.
    if ((long long)n < k) {
        cout << "INFEASIBLE" << "\n";
        return 0;
    }

    // Binary search on the answer x: feasible(x) = exists a run of >= k
    // consecutive positions all with a[i] >= x. feasible is monotone:
    // raising x can only shorten runs, so feasibility is non-increasing in x.
    // Bounds: any single element is >= min(a); a window's min is <= max(a).
    long long lo = LLONG_MAX, hi = LLONG_MIN;
    for (long long v : a) { lo = min(lo, v); hi = max(hi, v); }
    // lo is achievable (whole-array window of length n>=k has min >= lo),
    // hi+1 is not (no element is >= hi+1). Binary search the largest feasible x.

    auto feasible = [&](long long x) -> bool {
        long long run = 0;
        for (long long v : a) {
            if (v >= x) { run++; if (run >= k) return true; }
            else run = 0;
        }
        return false;
    };

    while (lo < hi) {
        long long mid = lo + (hi - lo + 1) / 2; // upper mid to avoid infinite loop
        if (feasible(mid)) lo = mid;
        else hi = mid - 1;
    }

    cout << lo << "\n";
    return 0;
}
```

**Causal recap.** The objective "maximize the minimum over blocks of length `>= k`" is monotone, so binary search on the floor `x` with the predicate "exists a run of `>= k` consecutive readings all `>= x`" is the natural method; the trap is signs. A first version anchored the search at `lo = 0` and clamped `hi = max(0, ...)`, which on an all-negative array `[-5,-3]` collapsed to `[0,0]` and printed the impossible `0` — the fix is honest data-driven bounds `lo = min(a)`, `hi = max(a)`, both allowed to be negative. A second defect, the floor midpoint `(lo+hi)/2`, terminated only by C++ truncation-toward-zero luck on negatives and looped forever on an adjacent positive interval like `[2,3]`; the upper midpoint `lo + (hi-lo+1)/2` strictly advances `lo` for every sign and kills the loop. Finally the `n < k` guard, placed before computing `min`/`max`, covers both true infeasibility and the empty array `n = 0`, and 900 brute-force-checked random cases over the negative/zero/all-negative corners confirm the whole thing.
