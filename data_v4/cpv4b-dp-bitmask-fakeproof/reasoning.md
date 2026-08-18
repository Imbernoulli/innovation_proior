Exact partition of an `N`-module set (`N <= 16`) by pairwise-disjoint crew masks, minimizing total hiring cost, `-1` if no partition exists. The recurrence is the textbook subset DP, so what decides this problem is two numbers I have to fix before writing a line. The accumulator width first: a partition uses up to `16` crews at `c[j] <= 10^9` each, so a total reaches `1.6*10^10` — past 32-bit, so every cost is `long long`. The second, and the actual trap, is the transition cost at `N = 16`: the per-mask loop runs over submasks, and the total submask work is exactly the kind of bit-counting sum a careless "average popcount" argument gets wrong by a factor that grows with `N`.

**Reducing the roster.** Many crews can carry the same mask at different costs, and the DP only ever wants the cheapest one, so I collapse to `cost1[s] = min c[j] over crews with mask[j] == s` (`INF` when none). This also disposes of the at-most-once constraint for free: along any partition the chosen masks are disjoint, so no mask repeats, so no crew is reused — the DP never has to track usage.

**The recurrence.** Let `best[m]` be the cheapest exact cover of set `m`, `best[0] = 0`. Every cover of `m` assigns the lowest set module `low` to exactly one crew, whose mask `s` is a submask of `m` containing `low`; the rest of `m` is covered by `best[m ^ s]`:

Let me confirm the recurrence by hand on the stated sample: `N = 4`, crews `(3,3) (12,4) (1,5) (6,2) (8,5) (15,11)`. So `cost1[3]=3` ({0,1}), `cost1[12]=4` ({2,3}), `cost1[1]=5` ({0}), `cost1[6]=2` ({1,2}), `cost1[8]=5` ({3}), `cost1[15]=11` (all). For `FULL = 15`, `low = bit 0`. Submasks of 15 containing bit 0 with a finite `cost1`: `s=1` (cost1 5, needs best[14]), `s=3` (cost1 3, needs best[12]), `s=15` (cost1 11, needs best[0]=0). I need `best[12]` and `best[14]`. `best[12]` = set {2,3}: its low is bit 2; submasks containing bit 2 with finite cost: `s=12` (cost1 4, best[0]=0) -> 4. So `best[12]=4`, giving the `s=3` branch `3 + 4 = 7`. The `s=15` branch gives `11 + 0 = 11`. `best[14]` = {1,2,3}: low is bit 1; submasks containing bit 1 with finite cost: `s=6` ({1,2}, cost1 2, needs best[8]); `best[8]` = {3} = `cost1[8]=5`, so that branch is `2 + 5 = 7`; any other? `s=2` and `s=14` have no crew. So `best[14]=7`, and the `s=1` branch of `best[15]` is `5 + 7 = 12`. Overall `best[15] = min(12, 7, 11) = 7`. Matches the documented answer `7`. The recurrence is right.

**Now the dangerous part: deriving the cost, and being honest that I am tempted to assert it.** The DP's running time is the total number of (mask, submask) pairs I touch, `Σ_{m=0}^{2^N-1} (number of submasks of m) = Σ_m 2^popcount(m)`. I need a closed form. My first instinct — the one I would happily write into a comment and never check — is to reason "by averages": there are `2^N` masks, the average popcount is `N/2`, so the average number of submasks is `2^(N/2)`, hence the total is about `2^N * 2^(N/2) = 2^(1.5 N)`. For `N = 16` that is `2^24 ≈ 1.7*10^7`, comfortably in budget, so I would be tempted to stop here.

But "average of `2^popcount`" is **not** `2^(average popcount)` — that step silently swaps `E[2^X]` for `2^E[X]`, and `2^X` is convex, so Jensen says `E[2^X] >= 2^E[X]`. My average argument therefore *under*estimates the work, and I do not yet know by how much. Before I rely on `2^(1.5N)` I am going to compute the sum exactly, derive its true closed form, and check both against each other numerically. This is exactly the kind of plausible-but-false bit-counting step that ships wrong solutions.

```
best[m] = min over submasks s of m with (s & low) and cost1[s] < INF of  cost1[s] + best[m ^ s]
```

Forcing `low ∈ s` counts each partition once instead of once per ordering of its parts — the minimum would be right either way, but the restriction also cuts the per-mask work down to submasks of `rest`. The answer is `best[2^N - 1]`.

**Sizing the transition — the trap.** The runtime is the total number of (mask, submask) pairs, `Σ_m 2^popcount(m)`. The instinct I would happily bake into a comment: `2^N` masks, average popcount `N/2`, so `≈ 2^N * 2^(N/2) = 2^(1.5N)`, which at `N=16` is `1.7*10^7` — comfortable, stop here. But that step swaps `E[2^X]` for `2^E[X]`, and `2^X` is convex, so by Jensen it *under*-counts, and I do not yet know by how much. Count the pairs directly instead: each bit is independently outside `m`, inside `m` but not `s`, or inside both — three states, independent across bits, so

```
Σ_m 2^popcount(m) = 3^N.
```

At `N=16` that is `3^16 ≈ 4.3*10^7`, about 2.5x the convexity-broken guess and still fine for 2 s, but the naive estimate was low in the dangerous direction — on a tighter limit that gap is a timeout. Summing directly at small `N` confirms the closed form and the sign of the error: `N=4` gives `81 = 3^4` versus `2^6 = 64`, `N=8` gives `6561 = 3^8` versus `4096`. So I size on `3^N`, not on the average-popcount formula.

This enumerates *all* nonempty submasks rather than only those containing `low`. The minimum is still correct (it just considers each part as "the split-off one" in every order), so I am not worried about wrong answers here — I am worried about a subtler defect, so let me trace a tiny case to watch the values. Take `N = 1`, one crew `(mask=1, c=5)`: `cost1[1]=5`, `best[0]=0`. `mask=1`: `sub=1`, `cost1[1]=5`, `best[0]=0` -> `bm = 5`. `best[1]=5`, answer `5`. Correct. Now `N = 2`, crews `(1,5) (2,5)` only (no crew for {0,1}): `cost1[1]=5, cost1[2]=5, cost1[3]=INF`. `best[1]=5`, `best[2]=5`. `mask=3`: submasks `3,2,1`. `sub=3`: `cost1[3]=INF`, skip. `sub=2`: `cost1[2]=5`, `best[1]=5` -> `bm=10`. `sub=1`: `cost1[1]=5`, `best[2]=5` -> `bm=10`. `best[3]=10`, answer `10`. The independent brute on the same input also says `10`. Correct, but I notice the all-submasks loop did the `{0}+{1}` split twice (as `sub=2` and `sub=1`) — wasted work, exactly the redundancy the `low`-restriction removes. Functionally fine, but at `N=16` I would rather not double-count, so I will switch to the lowbit-restricted enumeration in the final version.

**Second real bug: forgetting `N = 0`.** Trace the empty station: `N = 0`, `FULL = (1<<0) - 1 = 0`. My table `vector<long long> best(1 << 0)` has size `1`, `best[0] = 0`. The DP loop `for (mask = 1; mask <= FULL=0; ...)` never runs. Answer is `best[FULL] = best[0] = 0`. That is actually correct — the empty collection partitions the empty module set at cost `0`. But I almost wrote the cost1 table as `vector<long long> cost1(1 << n, INF)` and then, in an earlier draft, indexed `cost1[mk]` while reading crews *before* checking `mk <= FULL`; with `N = 0`, `FULL = 0`, any `mk >= 1` is out of range and `cost1[mk]` would be an out-of-bounds write into a size-1 vector. The contract says `mask[j] >= 1`, so for `N = 0` there should be no crews, but a defensive `if (mk >= 1 && mk <= FULL)` guard on the read makes a stray crew harmless. I add that guard, and I special-case the print for `N = 0` to emit `0` directly so there is no ambiguity. Re-trace `N=0, M=0`: prints `0`. Correct.

**Switching to the lowbit-restricted, verified loop.** Replacing the all-submasks loop with the lowbit version: for `mask`, take `low = mask & (-mask)`, `rest = mask ^ low`, and enumerate `s = low | (submask of rest)`. Re-trace `N=2`, crews `(1,5)(2,5)`: `mask=3`, `low=1`, `rest=2`. Submasks of rest: `2` then `0`. `sub=2 -> s=3`: `cost1[3]=INF`, skip. `sub=0 -> s=1`: `cost1[1]=5`, `best[3^1=2]=5` -> `bm = 10`. `best[3]=10`. Same answer `10`, but now the `{0}+{1}` partition is considered exactly once (as `s=1`, the part containing `low`). Re-trace the sample `N=4` `best[15]`: `low=1`, `rest=14`; submasks of 14 containing... the loop builds `s = 1 | sub` for `sub ⊆ {1,2,3}`, hitting `s=1` (best[14]), `s=3` (best[12]), `s=15` (best[0]), etc. — exactly the three finite branches I traced by hand, giving `7`. Consistent.

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
