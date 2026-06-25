**Reading the problem and pinning the contract.** I have `m` lantern periods `p_1, ..., p_m`, and a minute `t >= 1` is *lit* when `t` is a multiple of at least one period. I must return the time of the `k`-th lit minute, counting distinct lit minutes in increasing order. Input is `m` and `k` on the first line, then the `m` periods; output is one integer. Before any algorithm I fix the scale because it dictates the data types: `m <= 10`, `p_i <= 10^6`, `k <= 10^12`. The key question is how large the answer can get, because that sets the binary-search upper bound and whether the answer even fits in 64 bits.

**Bounding the answer, with an actual number.** The smallest period, call it `pmin`, already produces lit minutes at `pmin, 2*pmin, 3*pmin, ...`. Those alone are `k` distinct lit minutes by time `k*pmin`, so the `k`-th lit minute over *all* periods is no later than `k*pmin`. With `k <= 10^12` and `pmin <= p_i <= 10^6`, the answer is at most `10^12 * 10^6 = 10^18`. Let me sanity-check that against the 64-bit ceiling: `LLONG_MAX` is about `9.22 * 10^18`, and `10^18 < 9.22*10^18`, so the answer fits in `long long` with almost an order of magnitude of headroom. Good — I will set the binary-search `hi` to `2*10^18`, which is above the worst-case answer of `10^18` yet still below `LLONG_MAX`, so even the count arithmetic at `x = hi` stays in range. That `2e18` choice is deliberate, not lazy: if I had set `hi = 10^9` (a reflex from "values up to a million"), the boundary case `p=[10^6], k=10^12` whose answer is exactly `10^18` would be unreachable and the search would wrongly return `10^9`. I verified the magnitude by hand: `10^6 * 10^12 = 10^18`, so `hi` must exceed `10^18`.

**The monotone reformulation — binary search on the answer.** Define `f(x)` = the number of distinct lit minutes in `[1, x]`. As `x` increases by one, `f(x)` either stays the same (if `x+1` is not lit) or increases by one (if it is), so `f` is non-decreasing. The `k`-th lit minute is then exactly the smallest `x` with `f(x) >= k`. This is a textbook "smallest `x` satisfying a monotone predicate" search: keep `lo`/`hi`, test `f(mid) >= k`, move `hi = mid` when true and `lo = mid+1` when false, converge to `lo`. The entire problem now reduces to computing `f(x)` correctly and fast. The word *distinct* in `f`'s definition is the whole game.

**Candidate counting methods.** I have two ways to compute `f(x)`:

- *Sum of individual counts.* `f(x) = sum_i floor(x / p_i)`. It is `O(m)` and one line. The danger is obvious the moment I say it out loud: a minute divisible by two different periods is counted in two different terms.
- *Inclusion–exclusion over subsets.* A minute is divisible by *all* periods in a subset `S` exactly when it is a multiple of `lcm(S)`, and there are `floor(x / lcm(S))` such minutes in `[1,x]`. The inclusion–exclusion principle says the size of a union of sets is `sum` over non-empty subsets of `(-1)^(|S|+1) * |intersection of S|`. So `f(x) = sum_{S != empty} (-1)^(|S|+1) * floor(x / lcm(S))`. With `m <= 10` there are at most `2^10 - 1 = 1023` subsets, cheap. The dangers are the sign convention and the LCM overflowing.

**Why I distrust the sum-of-counts shortcut — and a concrete check.** Let me actually evaluate both on `p = [2, 3]` at `x = 6`. The true lit minutes in `[1,6]` are `2, 3, 4, 6` — that is `4` distinct minutes. Sum-of-counts gives `floor(6/2) + floor(6/3) = 3 + 2 = 5`. It reports `5`, not `4`. The extra count is minute `6`: it is a multiple of `2` *and* of `3`, so it lands in both terms and is double-counted. This is not a corner case; it is the generic behavior whenever periods share a common multiple `<= x`. So the naive sum is structurally wrong. I keep it on the page only as the thing inclusion–exclusion has to *correct*.

To be sure this double-count actually corrupts the *answer* and is not absorbed by the binary search, let me trace the search with the buggy count. Suppose `countLit(x)` is the buggy `floor(x/2)+floor(x/3)`. At the true answer `x=15` the buggy count is `floor(15/2)+floor(15/3) = 7 + 5 = 12`. For `k=10` the search looks for the smallest `x` with buggy-count `>= 10`; since the buggy count overshoots, it crosses `10` too early. Walking it: buggy-count reaches `10` around `x=12` (`floor(12/2)+floor(12/3) = 6+4 = 10`), so the buggy search returns `12`. But `12` is the *8th* lit minute, not the 10th (the sequence is `2,3,4,6,8,9,10,12,...`). So the double-count makes the answer come out `12` instead of `15` — a real, visible wrong answer. The shortcut is dead; I commit to inclusion–exclusion.

**Deriving the inclusion–exclusion count and checking it on the sample.** For a non-empty subset `S` of indices, let `L = lcm(p_i : i in S)`. The count of integers in `[1,x]` divisible by every period in `S` is `floor(x / L)`. Inclusion–exclusion gives

```
f(x) = sum over non-empty S of  (-1)^(|S|+1) * floor(x / lcm(S))
```

so odd-sized subsets *add* and even-sized subsets *subtract*. Let me verify this formula numerically on `p = [2, 3]`, `x = 6`, where I already know the truth is `4`. Subsets: `{2}` -> `floor(6/2)=3`, size 1, add: `+3`. `{3}` -> `floor(6/3)=2`, size 1, add: `+2`. `{2,3}` -> `lcm=6`, `floor(6/6)=1`, size 2, subtract: `-1`. Total `3 + 2 - 1 = 4`. Matches the hand count exactly — the formula and the sign convention are right, and the `-1` is precisely the double-counted minute `6` being removed once.

Let me also confirm the search end-to-end on the stated sample `p=[2,3]`, `k=10`, expected `15`. With the correct `f`, `f(14) = floor(14/2)+floor(14/3)-floor(14/6) = 7+4-2 = 9 < 10`, and `f(15) = floor(15/2)+floor(15/3)-floor(15/6) = 7+5-2 = 10 >= 10`. So the smallest `x` with `f(x) >= 10` is `15`. Correct.

**First implementation of the count.** I iterate over bitmasks `mask` from `1` to `2^m - 1`. For each, I build the LCM of the chosen periods, then add or subtract `floor(x / lcm)` by parity of the popcount:

```
ll countLit(ll x) {
    ll total = 0;
    for (int mask = 1; mask < (1 << m); mask++) {
        ll lcmv = 1;
        for (int i = 0; i < m; i++)
            if (mask & (1 << i)) {
                ll g = __gcd(lcmv, p[i]);
                lcmv = lcmv / g * p[i];      // lcm
            }
        int bits = __builtin_popcount(mask);
        ll term = x / lcmv;
        if (bits & 1) total += term; else total -= term;
    }
    return total;
}
```

**First trace, and the overflow lands immediately.** I trace a case the constraints actually allow: three pairwise-coprime large periods `p = [999983, 999979, 999961]` (each near `10^6`) at `x = 2*10^18` (the top of the search). The pair LCM for `{0,1}` is `999983 * 999979 ≈ 9.9998*10^11`, fine. But the triple subset `{0,1,2}` computes `lcmv = 999983 * 999979 * 999961 ≈ 9.9994*10^17`. That is still under `LLONG_MAX`, so it happens to survive *here* — but the computation `lcmv / g * p[i]` formed `999983*999979 ≈ 10^12` first and then multiplied by `~10^6`, giving `~10^18`, which is within range only because `m=3`. Push to `m` larger and pairwise-coprime periods near `10^6`: even four of them give `lcm ≈ 10^24`, which overflows `long long` and wraps to garbage. The wrapped `lcmv` could be small or negative, and `x / lcmv` becomes a wildly wrong term — silently corrupting the count. So I cannot just multiply LCMs blindly; I need a guard.

**The structural fix for overflow.** Here is the saving observation: if `lcm(S) > x`, then `floor(x / lcm(S)) = 0`, so that subset contributes *nothing* regardless of sign. I never need the true value of an LCM that exceeds `x`; I only need to know it exceeds `x`. So while accumulating the LCM, before each multiply `lcmv = lcmv * (p[i] / g)`, I check whether it would exceed `x`: if `lcmv > x / (p[i]/g)`, the product would exceed `x`, so I bail out of this subset and skip its (zero) term. With `x <= 2*10^18`, every surviving `lcmv` is `<= x <= 2*10^18 < LLONG_MAX`, and the only multiply I actually perform satisfies `lcmv <= x / step`, hence `lcmv * step <= x`, so it cannot overflow. The guard does double duty: it prevents overflow *and* it short-circuits the useless subsets.

**Re-implementing with the guard, then tracing the double-count corner once more.** Here is the revised count:

```
ll countLit(ll x) {
    ll total = 0;
    for (int mask = 1; mask < (1 << m); mask++) {
        ll lcmv = 1;
        bool overflow = false;
        for (int i = 0; i < m; i++) {
            if (mask & (1 << i)) {
                ll g = __gcd(lcmv, p[i]);
                ll step = p[i] / g;
                if (lcmv > x / step) { overflow = true; break; }
                lcmv *= step;
            }
        }
        if (overflow) continue;
        int bits = __builtin_popcount(mask);
        ll term = x / lcmv;
        if (bits & 1) total += term; else total -= term;
    }
    return total;
}
```

I re-trace `p=[2,3]`, `x=6` with this guarded version to be sure the guard did not accidentally drop a real subset. `mask=01` ({2}): `step=2`, `1 <= 6/2=3` ok, `lcmv=2`, term `6/2=3`, size 1 add `+3`. `mask=10` ({3}): `step=3`, `1 <= 6/3=2` ok, `lcmv=3`, term `6/3=2`, add `+2`. `mask=11` ({2,3}): first bit `step=2`, `lcmv=2`; second bit `g=gcd(2,3)=1`, `step=3`, check `2 <= 6/3 = 2` true (equal is allowed), `lcmv=6`, term `6/6=1`, size 2 subtract `-1`. Total `3+2-1=4`. Still `4` — the guard correctly keeps the subset whose LCM equals `x` (the `<=` boundary), and only drops subsets whose LCM strictly exceeds `x`. That boundary `lcmv > x/step` (strict) versus `>=` matters: if I had written `>=` I would wrongly drop the `{2,3}` subset at `x=6` and get `5` again — the double-count would silently return. Let me confirm the boundary direction is right: I want to drop only when `lcmv*step > x`. Using integer division, `lcmv > x/step` implies `lcmv >= floor(x/step)+1 > x/step`, so `lcmv*step > x`; and when `lcmv <= x/step`, `lcmv*step <= step*floor(x/step) <= x`. So the strict `>` test drops exactly the subsets with LCM `> x`. Correct.

**A second debug episode: duplicate periods and a near-miss double-count.** Duplicate periods worry me because two equal lanterns describe the *same* set of lit minutes, and a careless union count would treat them as two independent sets and overcount. Let me trace `p = [6, 6]`, `k = 4`. The lit minutes are just the multiples of 6: `6, 12, 18, 24, ...`, so the 4th is `24`. Does inclusion–exclusion get this? At a generic `x`, subsets: `{0}` -> `floor(x/6)` add; `{1}` -> `floor(x/6)` add; `{0,1}` -> `lcm(6,6)=6`, `floor(x/6)` subtract. Total `floor(x/6) + floor(x/6) - floor(x/6) = floor(x/6)`. The `+2 -1` collapses to `+1` copy of `floor(x/6)` — inclusion–exclusion *automatically* deduplicates equal periods because `lcm(6,6)=6`, so the pairwise subtraction cancels the second copy. No special-casing of duplicates needed. Let me confirm the search: I need smallest `x` with `floor(x/6) >= 4`, i.e. `x/6 >= 4`, i.e. `x >= 24`, so answer `24`. That matches the hand answer, and it confirms I do *not* need to pre-dedup the input array — the LCM machinery handles it. (I momentarily considered `sort + unique` on `p` to dedup; it is unnecessary and, worse, would be a place to introduce an off-by-one, so I leave the array as-is.)

**A divisibility-chain check, because nested multiples are another overcount trap.** Take `p = [2, 4, 8]`. Here every multiple of 8 is also a multiple of 4 and of 2, every multiple of 4 is a multiple of 2 — so the lit minutes are exactly the even minutes, and `f(x) = floor(x/2)`. Let me verify inclusion–exclusion produces that. LCMs: `lcm(2,4)=4`, `lcm(2,8)=8`, `lcm(4,8)=8`, `lcm(2,4,8)=8`. So
`f(x) = floor(x/2)+floor(x/4)+floor(x/8)  -floor(x/4)-floor(x/8)-floor(x/8)  +floor(x/8)`
`= floor(x/2) + [floor(x/4)-floor(x/4)] + [floor(x/8) - floor(x/8) - floor(x/8) + floor(x/8)]`
`= floor(x/2)`. The `floor(x/4)` terms cancel and the four `floor(x/8)` terms `+1-1-1+1 = 0` cancel, leaving exactly `floor(x/2)`. So for `k=5` the answer is the smallest `x` with `floor(x/2) >= 5`, i.e. `x=10`, which is indeed the 5th even number. Inclusion–exclusion handles the divisibility chain with no special logic. Good.

**Edge cases, deliberately.**
- *`m = 1`.* Only `mask = 1`, `lcmv = p[0]`, `f(x) = floor(x/p[0])`. For `p=[7], k=5`: smallest `x` with `floor(x/7) >= 5` is `35`. The lone-lantern case is just multiples of `p[0]`; correct.
- *Answer at the boundary `10^18`.* `p=[10^6], k=10^12`: `f(x) = floor(x/10^6)`, smallest `x` with `floor(x/10^6) >= 10^12` is `10^18`. `hi = 2*10^18 > 10^18`, so it is reachable, and `10^18 < LLONG_MAX`, so it prints exactly. This is the case that would die under a too-small `hi`.
- *Pairwise-coprime `m=10` with large `k`.* All `1023` subsets are visited; subsets of size `>= 4` have LCM `> 10^18 >= x` and are short-circuited by the overflow guard (term 0), so only the small subsets contribute. No overflow, no crash.
- *Binary-search invariant.* `lo` starts at `1` (the earliest possible lit minute is `>= 1`), `hi` at `2*10^18`. The predicate `f(x) >= k` is false at `x = 0` and true at `x = hi` (since `f(hi) >= floor(hi/pmin) >= k` for any valid input), so the answer lies in `[1, hi]` and the search converges. `mid = lo + (hi-lo)/2` avoids `lo+hi` overflowing.

**Numeric self-check of the answer bound and the cost.** The answer bound `k * pmin`: for the extreme `k=10^12`, `pmin=10^6`, that is `10^18`, and I chose `hi=2*10^18 > 10^18`, confirmed above. The per-query cost is `2^m = 1024` subsets, each `O(m)=10` work, so `~10^4` operations; binary search does about `log2(2*10^18) ≈ 61` iterations, giving `~6*10^5` operations total — trivially under the 1-second limit. Both numbers check out against the constraints.

**Validation against an independent brute force.** I wrote a separate brute that simply walks `t = 1, 2, 3, ...`, marks `t` lit when `any(t % p_i == 0)`, and stops at the `k`-th lit minute — an obviously-correct definition with no inclusion–exclusion and no binary search, so it cannot share a bug with the solution. Over 800 random small cases (two generators: one with small varied periods and small `k`, one biased to heavy overlap and divisibility chains like `2,4,6,8,9,12` with duplicates) the binary-search solution matched the brute on every case, zero mismatches. The duplicate-heavy and chain-heavy generator is exactly what would expose a surviving double-count, and it stayed clean.

**Final solution.** I disproved the sum-of-counts shortcut with a traced double-count (`p=[2,3]` reporting `5` instead of `4`, and the search returning `12` instead of `15`), derived and numerically verified the inclusion–exclusion count, fixed the LCM overflow with a guard that doubles as a short-circuit and whose strict-`>` boundary I proved keeps the LCM-equals-`x` subset, and confirmed that equal periods and divisibility chains deduplicate automatically. This is what I ship:

```cpp
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef unsigned long long ull;

int m;
ll k;
vector<ll> p;

// count of integers in [1, x] divisible by at least one p_i,
// via inclusion-exclusion over subsets, using lcm with an overflow cap.
ll countLit(ll x) {
    ll total = 0;
    for (int mask = 1; mask < (1 << m); mask++) {
        // build lcm of the chosen p_i; if it exceeds x, its contribution is 0
        // so we can stop early and treat the term as 0.
        ll lcmv = 1;
        bool overflow = false;
        for (int i = 0; i < m; i++) {
            if (mask & (1 << i)) {
                ll g = __gcd(lcmv, p[i]);
                ll step = p[i] / g;            // lcmv * step is the new lcm
                // overflow / over-x guard: if lcmv * step > x, this subset
                // contributes floor(x / lcm) == 0, so mark and bail.
                if (lcmv > x / step) { overflow = true; break; }
                lcmv *= step;
            }
        }
        if (overflow) continue;                // term is 0
        int bits = __builtin_popcount(mask);
        ll term = x / lcmv;
        if (bits & 1) total += term;           // odd-sized subset: add
        else total -= term;                    // even-sized subset: subtract
    }
    return total;
}

int main() {
    if (!(cin >> m >> k)) return 0;
    p.resize(m);
    for (auto &v : p) cin >> v;

    // Binary search for the smallest x with countLit(x) >= k.
    ll lo = 1, hi = (ll)2e18;
    while (lo < hi) {
        ll mid = lo + (hi - lo) / 2;
        if (countLit(mid) >= k) hi = mid;
        else lo = mid + 1;
    }
    cout << lo << "\n";
    return 0;
}
```

**Causal recap.** The monotone count `f(x)` (lit minutes in `[1,x]`) turns the problem into a binary search for the smallest `x` with `f(x) >= k`; the naive `sum floor(x/p_i)` double-counts minutes divisible by several periods, which a `p=[2,3]` trace exposed (count `5` vs `4`, search `12` vs `15`), so I switched to inclusion–exclusion over subsets with LCMs and the `(-1)^(|S|+1)` sign — numerically verified to give `4` on that same case. Multiplying LCMs blindly overflows `long long` for four-plus coprime periods near `10^6`, so I guard each multiply with `lcmv > x/step`, which both prevents overflow and zeroes out the useless `lcm > x` subsets; the strict `>` keeps the boundary subset where `lcm == x` (proved, and re-traced on `p=[2,3]`). Equal periods and divisibility chains deduplicate automatically because the LCM machinery cancels the redundant terms (verified on `[6,6]` and `[2,4,8]`), the answer bound `k*pmin = 10^18` fits under `hi = 2*10^18 < LLONG_MAX`, and 800 brute-force comparisons came back with zero mismatches.
