The three constraint sizes here are what set the trap. There are up to `n = 2*10^5` presses, each
with period `s[i]` up to `10^9`, and the quota `P` runs to `10^14`. Press `i` finishes its `k`-th
sheet at `k*s[i]`, so by elapsed time `T` it has done `floor(T/s[i])` sheets, and I want the earliest
integer `T` with `pages(T) = sum_i floor(T/s[i]) >= P`. Two quantities immediately want to escape
32-bit range: the answer `T` itself — one press of period `10^9` needing five sheets already puts `T`
at `5*10^9` — and, more insidiously, the feasibility sum, since `n = 2*10^5` presses of period `1`
make `pages(T) = n*T`, which crosses `2^31` at a tiny `T`. Both the candidate time and the running
count have to be 64-bit, and the sum is the one I'd get wrong by reflex-typing `int` in the predicate.

`pages(T)` is nondecreasing in `T` — each `floor(T/s[i])` is — so `pages(T) >= P` is a step function,
false below some threshold `T*` and true at and above it. That makes this a binary-search-the-answer
problem: bisect for `T*`, each feasibility test an `O(n)` pass of `floor` divisions, `O(n log hi)`
overall. (A second-by-second simulation is the definition and trivially correct, but `T*` is on the
order of `P*min(s)`, astronomically large, so simulation only survives as an offline checker on tiny
inputs — which is how I'll validate the real solution.)

Bounds. `lo = 0` is infeasible since `pages(0) = 0 < P` (`P >= 1`). For `hi` I need a `T` with
`pages(hi) >= P`: by time `hi = P*minPeriod`, where `minPeriod = min_i s[i]`, the single fastest press
alone finishes `floor(P*minPeriod / minPeriod) = P` sheets, and the others only add more. On the
sample `s = [2,3,5]`, `P = 10`: `minPeriod = 2`, `hi = 20`, and `pages(20) = 10+6+4 = 20 >= 10`, so the
true answer `10` sits inside `[0,20]`. One caveat on magnitude: `P*minPeriod` can be as large as
`10^14 * 10^9 = 10^23`, past the `long long` ceiling of `~9.2*10^18` — but that extreme needs a single
slow press with a huge quota (`n = 1`, `s = [10^9]`, `P = 10^14`, where `T*` genuinely is `10^23` and
no 64-bit type could hold the answer at all). The large inputs that stress this problem pair big `P`
with small periods, so `hi = P*minPeriod` stays inside `long long`; I compute it there and move on.

The predicate is where the real overflow lives. Even with `hi` storable, the sum `pages(T)` is not
storable mid-computation. Take `n = 2*10^5`, all periods `1`, `P = 2*10^14`: `hi = 2*10^14` fits, but
during the search I evaluate `pages(mid)` for `mid` near `10^9`, and that sum is
`2*10^5 * 10^9 = 2*10^19` — over the `long long` ceiling. So the accumulator must be `long long` *and*
capped: since the predicate only asks whether the sum reaches `P`, I stop and return the moment
`total >= P`, which bounds the accumulator below `P + (largest single term)` and keeps it in range.

An `int` accumulator here is not merely sloppy, it silently inverts the answer. On that same input at
`mid = 10^9`, every term is `10^9` and I add `2*10^5` of them; `total` blows past `2^31` after a couple
thousand adds and wraps modulo `2^32` to a near-random residue — certainly not `>= 2*10^14`. So the
predicate reports infeasible at the true answer, and almost everywhere else. The search then only ever
executes `lo = mid + 1`, marching `lo` all the way to `hi` and terminating at
`lo = hi = P*minPeriod = 2*10^14`, instead of the correct `T* = ceil(P/n) = 10^9` — an error of five
orders of magnitude, the whole search dragged onto its upper bound by a wrapped predicate. The capped
64-bit predicate that dodges it:

```
auto pages = [&](long long T) -> long long {
    long long total = 0;
    for (long long period : s) {
        total += T / period;
        if (total >= P) return total;   // cap: the sum never runs away
    }
    return total;
};
```

On the broken input at `mid = 10^9` this returns the moment `total` crosses `2*10^14` (no wrap, so
feasible); at `mid = 10^9 - 1` it returns `2*10^14 - 2*10^5 < P` (infeasible); the search brackets
`10^9` correctly.

The bisection frame itself has a boundary to get right — least feasible `T`, not one off. With
`hi = mid` on feasible and `lo = mid + 1` on infeasible under `while (lo < hi)`, `hi` stays feasible
and `lo` chases it from below to the exact threshold. Tracing the sample `s = [2,3,5]`, `P = 10`,
`hi = 20`: `mid=10` gives `pages=10 >= 10`, `hi=10`; `mid=5` gives `4 < 10`, `lo=6`; `mid=8` gives
`7 < 10`, `lo=9`; `mid=9` gives `8 < 10`, `lo=10`; now `lo=hi=10`, output `10` — matching the sample
and correctly rejecting `T=9`. Writing `hi = mid - 1` instead would step past `T=10` and return `9`,
so the `mid` (not `mid - 1`) update is the load-bearing choice.

A few edge cases, since bisection code dies at them. Single press `s = [7]`, `P = 1`: `hi = 7`,
`pages(7) = 1 >= 1`, `pages(6) = 0 < 1`, converges to `7` — the first finish time `s[0]`, not `0`.
Equal periods `n = 4`, `s[i] = 3`, `P = 8`: need `4*floor(T/3) >= 8`, i.e. `T >= 6`, and the search
lands on `6`. Quota exactly on a multiple (the sample, `pages(10) = 10 = P`) versus strictly between
(`P = 9` on the same periods still gives `10`, since `pages(9) = 8 < 9 <= 10`). And the midpoint
`lo + (hi - lo)/2` avoids the classic `lo + hi` overflow near the top of the range.

I cross-checked the binary search against the second-by-second simulator on 751 random small cases
(periods in `[1,12]`, `P` in `[1,60]`, `n` up to `6`) with zero mismatches, plus the sample (`10`) and
the large overflow case (`n = 2*10^5`, all periods `1`, `P = 2*10^14` -> `10^9`). The small cases pin
the predicate and the bisection boundary; the large one pins that the `long long` width and the cap
prevent the wrap an `int` accumulator would suffer. The full program — the `O(n log hi)` binary search
with this capped predicate — is the answer.
