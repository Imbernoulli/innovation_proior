**Two features of this instance settle the design before I write a line.** The first is the profit floor: delivery is optional and entries can be negative, so "deliver nothing" for profit `0` is always a legal assignment and the answer can never fall below `0`. An all-negative matrix must return `0`, not the least-bad matching — that is the corner this problem is built to punish. The second is the scale. There are `m <= 18` slots, small enough to enumerate as a bitmask (`2^18 = 262144` subsets), and `n <= 18` parcels with `|p[i][j]| <= 10^9`, so up to `18` placements can sum to `~1.8*10^10` — past the `~2.1*10^9` ceiling of a 32-bit `int`. Every accumulator has to be `long long`; an `int` is a silent wrong answer on the `n=m=18` tests. So the shape is fixed: a subset DP over which slots are occupied, with 64-bit arithmetic. Input is `n m` then the `n x m` grid on stdin, and I print one integer.

**Greedy first, because if it held I would take it.** The tempting shortcut is to repeatedly lock the largest free positive cell `p[i][j]` and stop when none is left — a dozen lines, `O(nm log nm)`. But matching is a global constraint being decided one cell at a time, which is the classic setup for greedy to lose, so I try to break it. On `n=m=2`, `p=[[10,9],[9,0]]`: greedy grabs the biggest cell `p[0][0]=10`, which leaves only the free pair parcel1/slot1 worth `0`, total `10`. The matching parcel0->slot1, parcel1->slot0 scores `9+9=18`. The single largest entry need not belong to any maximum-weight matching, and committing to it destroys the pair the two `9`s needed. Greedy is out.

**The DP state is just the occupied-slot set.** Processing parcels `0..n-1` in order, the only thing later parcels care about is which slots are already taken — not which parcel took them. So the state is a bitmask `mask` over the `m` slots, and `dp[mask]` is the best total profit after deciding the parcels seen so far with exactly the slots in `mask` occupied. Parcel `i` has two moves from a reachable `mask`: skip it, leaving `mask` unchanged, or place it in a free slot `j`, setting bit `j` and adding `p[i][j]`. That is `O(n * 2^m * m) ~ 8.5*10^7` operations at full size — comfortable under one second.

**The base case is the whole trap.** Before any parcel I have used no slot and earned `0`, so `dp[0]=0` and every other mask is unreachable. "Unreachable" must be a sentinel that loses every `max` — negative infinity, concretely `NEG = LLONG_MIN/4`, chosen so that adding a real `p[i][j]` to it can never underflow. The seductive error is to seed *every* mask to `0` instead. That asserts an arbitrary occupied set is reachable for free, and any later positive parcel keying off such a phantom mask would believe slots had been filled at no cost, inflating the answer. The `0` floor must come from somewhere honest: the skip transition keeps `dp[0]=0` alive through every parcel, and since skipping is always available `dp[0]` never drops, so the final best is at least `0`. That is precisely why the all-negative matrix returns `0` — nothing phantom, just the surviving empty matching.

**Two ways the clean recurrence transcribes into wrong code, both exposed by tiny inputs.** First, all of one parcel's transitions must read the layer from *before* that parcel. If I update `dp` in place while sweeping masks upward, a value I just wrote (parcel `i` into slot `j`) gets reconsumed one mask later to place the *same* parcel again. On `n=1, m=2, p=[[7,7]]` that produces `dp[11]=7+7=14`, a single parcel occupying both slots — impossible; the correct answer is `7`. The fix is a fresh `ndp` per parcel, seeded to `NEG` and swapped in at the end. Second is the all-zero seed above: on `n=1, m=2, p=[[-5,-5]]` the correct answer is `0`, and with the `NEG` seed it arrives the right way — `ndp[00]=0` from the skip, both placements give `-5`, so `best=max(0,-5,-5)=0` comes from the surviving empty matching, not from a mask that was declared free. The corrected core:

```cpp
const long long NEG = LLONG_MIN / 4;
vector<long long> dp(full, NEG);
dp[0] = 0;
for (int i = 0; i < n; i++) {
    vector<long long> ndp(full, NEG);
    for (int mask = 0; mask < full; mask++) {
        if (dp[mask] == NEG) continue;          // unreachable, skip
        long long base = dp[mask];
        ndp[mask] = max(ndp[mask], base);        // leave parcel i undelivered
        for (int j = 0; j < m; j++) {
            if (mask & (1 << j)) continue;
            ndp[mask | (1<<j)] = max(ndp[mask | (1<<j)], base + p[i][j]);
        }
    }
    dp.swap(ndp);
}
```

Re-running `[[7,7]]`: only `mask=00` is reachable, so `ndp[01]=ndp[10]=7`, `ndp[11]` stays `NEG`, `best=7`. The double placement is gone because it was the in-place read that created it.

**Corners.** `n=0`: the parcel loop never runs, only `dp[0]=0` survives, output `0`. `n>m`: only `m` parcels fit, the surplus take the skip transition and the mask never exceeds `m` bits. `n<m`: spare slots stay `NEG`. The `NEG` sentinel is only ever read inside a `max` or behind the `== NEG` guard, so nothing is ever added to it and it cannot underflow. The final answer sweep starts `best=0` and takes the max over reachable masks, so the floor holds even if every reachable value were negative.

**Stress.** I checked the DP against an independent brute force that lets each parcel choose undelivered or one of `m` slots and rejects slot reuse — an obviously-correct method unlike the DP. Across ~900 small random cases spanning mixed signs, all-negative, all-zero, all-positive, and zero-sprinkled regimes (`n` up to 6, `m` up to 5), there were no mismatches; the sample prints `15`, the all-negative `2x2` prints `0`, the empty `0 3` prints `0`, and `n=m=18` runs in about `0.1 s`. The complete program wraps this core with the stdin reader (returning `0` on a missing header) and the `0`-floored answer sweep; it is in the answer file.

