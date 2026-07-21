Exact partition of an `N`-module set (`N <= 16`) by pairwise-disjoint crew masks, minimizing total hiring cost, `-1` if no partition exists. The recurrence is the textbook subset DP, so what decides this problem is two numbers I have to fix before writing a line. The accumulator width first: a partition uses up to `16` crews at `c[j] <= 10^9` each, so a total reaches `1.6*10^10` — past 32-bit, so every cost is `long long`. The second, and the actual trap, is the transition cost at `N = 16`: the per-mask loop runs over submasks, and the total submask work is exactly the kind of bit-counting sum a careless "average popcount" argument gets wrong by a factor that grows with `N`.

**Reducing the roster.** Many crews can carry the same mask at different costs, and the DP only ever wants the cheapest one, so I collapse to `cost1[s] = min c[j] over crews with mask[j] == s` (`INF` when none). This also disposes of the at-most-once constraint for free: along any partition the chosen masks are disjoint, so no mask repeats, so no crew is reused — the DP never has to track usage.

**The recurrence.** Let `best[m]` be the cheapest exact cover of set `m`, `best[0] = 0`. Every cover of `m` assigns the lowest set module `low` to exactly one crew, whose mask `s` is a submask of `m` containing `low`; the rest of `m` is covered by `best[m ^ s]`:

```
best[m] = min over submasks s of m with (s & low) and cost1[s] < INF of  cost1[s] + best[m ^ s]
```

Forcing `low ∈ s` counts each partition once instead of once per ordering of its parts — the minimum would be right either way, but the restriction also cuts the per-mask work down to submasks of `rest`. The answer is `best[2^N - 1]`.

**Sizing the transition — the trap.** The runtime is the total number of (mask, submask) pairs, `Σ_m 2^popcount(m)`. The instinct I would happily bake into a comment: `2^N` masks, average popcount `N/2`, so `≈ 2^N * 2^(N/2) = 2^(1.5N)`, which at `N=16` is `1.7*10^7` — comfortable, stop here. But that step swaps `E[2^X]` for `2^E[X]`, and `2^X` is convex, so by Jensen it *under*-counts, and I do not yet know by how much. Count the pairs directly instead: each bit is independently outside `m`, inside `m` but not `s`, or inside both — three states, independent across bits, so

```
Σ_m 2^popcount(m) = 3^N.
```

At `N=16` that is `3^16 ≈ 4.3*10^7`, about 2.5x the convexity-broken guess and still fine for 2 s, but the naive estimate was low in the dangerous direction — on a tighter limit that gap is a timeout. Summing directly at small `N` confirms the closed form and the sign of the error: `N=4` gives `81 = 3^4` versus `2^6 = 64`, `N=8` gives `6561 = 3^8` versus `4096`. So I size on `3^N`, not on the average-popcount formula.

**Implementation, and a sentinel landmine.** First cut of the loop, over all nonempty submasks:

```
for (int mask = 1; mask <= FULL; mask++) {
    long long bm = INF;
    for (int sub = mask; sub > 0; sub = (sub - 1) & mask)
        if (cost1[sub] < INF && best[mask ^ sub] < INF)
            bm = min(bm, cost1[sub] + best[mask ^ sub]);
    best[mask] = bm;
}
```

The `< INF` guards on *both* terms are load-bearing. With `INF = 4e18` and an uncoverable submask, `cost1[s] + best[m^s]` adds a live sentinel: even where it does not overflow `long long`, the result is a huge finite number that competes in the `min` and lets a genuinely impossible mask return a bogus cost instead of staying `INF` — and chained across masks such values can drift below the final `>= INF` test and escape as a fake answer. Guarding `best[m^s] < INF` before the add keeps `best[m]` exactly `INF` for every uncoverable `m`. On the impossible roster `N=2`, only `(1,5)`: `best[2]` stays `INF` (no crew covers `{1}`), the `s=1` branch is skipped, `best[3]=INF` → prints `-1`.

This all-submasks loop is already the `3^N` count above, but it double-counts every partition once per ordering of its parts — harmless for correctness, wasteful in constant factor. The shipped loop fixes the lowest bit: `low = mask & -mask`, `rest = mask ^ low`, and enumerates `s = low | (submask of rest)`, so each partition is considered exactly once, via the split whose part contains `low`. That halves the submask work (per mask, `2^popcount(rest)` instead of `2^popcount(m)`) while staying `O(3^N)`.

**`N = 0`.** `FULL = (1<<0) - 1 = 0`; the empty collection partitions the empty set at cost `0`, and the DP would return `best[0] = 0` anyway. Two hazards live here: a stray crew with `mask >= 1` (the contract forbids it, but a `mk <= FULL` guard on the read stops an out-of-bounds write into the size-1 `cost1`), and routing the answer through the general `>= INF` path. I guard the read and special-case the print to emit `0` directly.

**Edge cases and stress.** Duplicate masks collapse to the cheapest via `cost1 = min(...)`; zero-cost crews satisfy `< INF` and participate normally; a singletons-only roster is forced to take every singleton. Checked against a recursive exact-cover oracle on random small instances — impossible rosters, duplicates, singletons — with no mismatches, and the dense `N=16` case (all `2^16 - 1` masks present) runs inside the limit, as the `3^N` count predicts.

The full submask-DP module is in the answer.
