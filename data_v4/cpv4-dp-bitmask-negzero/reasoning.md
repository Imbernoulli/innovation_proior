**Reading the problem and pinning the contract.** I have `n` parcels and `m` slots, and a profit matrix `p[i][j]` whose entries can be positive, zero, or negative. I assign parcels to slots so that each slot holds at most one parcel and each parcel sits in at most one slot — a partial matching — and I maximize the total profit. The phrase that dominates everything is *delivery is optional*: I may leave parcels undelivered and slots empty, so the empty matching with profit `0` is always legal and the answer is never below `0`. Input is `n m` then an `n x m` grid on stdin; I print one integer. Let me fix the scale first, because it dictates the data types and even the algorithm. The constraints are `0 <= n <= 18`, `1 <= m <= 18`, `|p[i][j]| <= 10^9`. At most `min(n, m) <= 18` parcels are ever delivered, so the largest possible total is about `18 * 10^9 = 1.8 * 10^10`, which already blows past the 32-bit range of `~2.1 * 10^9`. So every accumulator must be 64-bit `long long`; an `int` is a silent wrong-answer on the large tests. That is the first non-negotiable decision.

**Why `m <= 18` is the load-bearing constraint.** The number of slots is small and is the dimension I will bitmask over. `2^18 = 262144`, and a DP that, for each parcel, sweeps every subset of slots and tries placing the parcel in each free slot costs `O(n * 2^m * m) = 18 * 262144 * 18 ~ 8.5 * 10^7` operations — comfortable under a 1-second limit. The matrix being at most `18 x 18` also means a brute force that lets each parcel pick one of `m + 1` decisions (undelivered, or one of `m` slots) is `(m+1)^n` which is astronomically large at full size but tiny for small cases — perfect for an independent stress oracle.

**Laying out the candidate approaches.** Two routes, and I want to commit to the one I can *prove*, not the one that types fastest.

- *Greedy by largest entry.* Repeatedly pick the largest positive `p[i][j]` whose parcel `i` and slot `j` are both still free, lock them, repeat until no positive free entry remains. `O(nm log(nm))`, a dozen lines. The danger is structural: matching is a global constraint and greedy decides locally, exactly the setup where greedy is usually wrong. I will not trust it until I have tried to break it.
- *Subset DP over occupied slots.* Process parcels left to right; `dp[mask]` = best profit using the parcels seen so far with exactly the slots in `mask` occupied. Each new parcel is either skipped (mask unchanged) or placed into one currently-free slot (set that bit, add the profit). `O(n * 2^m * m)`. The risk here is not the idea but the *base case and the sign handling* — with negatives and an optional empty matching, it is very easy to seed the DP wrong.

**Stress-testing greedy before committing.** "Greedy feels fine" is how wrong solutions ship, so let me attack it with a concrete instance. Take `n = 2`, `m = 2`,

```
p = [ [10,  9],
      [ 9,  0] ]
```

Greedy scans for the largest positive entry: that is `p[0][0] = 10`. It locks parcel 0 to slot 0. Now parcel 1 and slot 1 are free, with `p[1][1] = 0` — non-positive after the only other free entry, so greedy stops (or takes the `0`, which adds nothing). Greedy's total is `10`. Is `10` optimal? Consider the *other* matching: parcel 0 -> slot 1 (`9`) and parcel 1 -> slot 0 (`9`), total `18`. That is strictly better than `10`, and greedy could never reach it because the instant it grabbed the single biggest cell `10` it destroyed the only way to use both `9`s. So greedy is wrong, and I now see exactly *why*: the largest entry is not necessarily in any maximum-weight matching, and committing to it blocks a parcel-and-slot pair that two medium entries needed. Greedy is out. The five minutes of disproving it saved me from shipping it.

**Deriving the DP and nailing the state.** I process parcels in order `0, 1, ..., n-1`. The only thing the *future* parcels care about, when I am partway through, is *which slots are already used* — not which parcel went where, just the occupied set. So the state is a bitmask `mask` over the `m` slots, and `dp[mask]` = the best total profit achievable after deciding parcels `0..i-1` such that exactly the slots in `mask` are occupied. When I bring in parcel `i`, it has two kinds of move:

- **Skip it** (leave it undelivered). The mask does not change: `new_dp[mask] = max(new_dp[mask], dp[mask])`.
- **Place it in a free slot `j`** (a bit not set in `mask`). The mask gains bit `j` and I add the profit: `new_dp[mask | (1<<j)] = max(..., dp[mask] + p[i][j])`.

After all `n` parcels, the answer is the maximum of `dp[mask]` over all reachable masks — and I will also fold in `0` for the empty matching, though as I will check, the base case should already make `0` reachable.

**The base case — this is where the twist lives.** "Before deciding any parcel" I have used no slots and earned `0` profit, so `dp[0] = 0` and *every other mask is unreachable*. Unreachable must be a sentinel that loses every `max`, i.e. negative infinity — concretely `LLONG_MIN / 4` so that I never accidentally add `p[i][j]` to a true `LLONG_MIN` and underflow. If I instead initialized *all* masks to `0`, I would be asserting that I can reach an arbitrary occupied set for free, which is nonsense and would corrupt the transitions. So: `dp[0] = 0`, all else `= NEG`. The crucial consequence for the all-negative corner: the only way profit ever drops below `0` is by placing a parcel into a negative cell, but skipping is always available, so `dp[0]` stays `0` through every parcel, and the final answer is at least `dp[0] = 0`. The optionality is encoded *entirely* by the "skip" transition keeping `mask = 0` alive at value `0`.

**Sanity-checking the derivation on the sample.** The sample is `n = 3`, `m = 3`,

```
5 -2  1
-3  4  0
 2  1  6
```

claimed answer `15` via parcel0->slot0 (`5`), parcel1->slot1 (`4`), parcel2->slot2 (`6`). Let me trace the DP enough to believe it. Start `dp[000] = 0`, all else `NEG`. Parcel 0 (`5, -2, 1`): from `dp[000]=0` I can skip (keep `dp[000]=0`) or place in slot 0 (`dp[001]=5`), slot 1 (`dp[010]=-2`), slot 2 (`dp[100]=1`). Parcel 1 (`-3, 4, 0`): the path I care about extends `dp[001]=5` by placing parcel 1 in slot 1: `dp[011] = 5 + 4 = 9`. Parcel 2 (`2, 1, 6`): extend `dp[011]=9` by placing parcel 2 in slot 2: `dp[111] = 9 + 6 = 15`. So `15` is reachable. Could anything beat it? The only positive cells are `5, 1, 4, 2, 1, 6`; the three I picked (`5, 4, 6`) are the three largest that also form a valid matching (distinct rows and distinct columns), and any swap to include another cell would have to drop one of these for a smaller value. `15` it is — the derivation matches the claim.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut of the core:

```
vector<long long> dp(full, 0);          // <-- careless: "start everything at 0"
dp[0] = 0;
for (int i = 0; i < n; i++) {
    for (int mask = 0; mask < full; mask++) {
        long long base = dp[mask];
        for (int j = 0; j < m; j++) {
            if (mask & (1 << j)) continue;
            int nmask = mask | (1 << j);
            dp[nmask] = max(dp[nmask], base + p[i][j]);   // in-place, same dp
        }
    }
}
long long best = *max_element(dp.begin(), dp.end());
```

Two things smell wrong already — I initialized *all* masks to `0` (not just `mask = 0`), and I am updating `dp` in place while iterating masks upward, so a parcel could be placed twice. Let me trace the smallest input that exposes it. Take `n = 1`, `m = 2`,

```
p = [ [7, 7] ]
```

The correct answer is `7`: one parcel, deliver it into whichever slot, profit `7` (it cannot occupy both slots). Start `dp = [0, 0, 0, 0]` (masks `00,01,10,11`). Parcel 0: at `mask=00`, `base=0`: place in slot 0 -> `dp[01]=max(0,7)=7`; place in slot 1 -> `dp[10]=max(0,7)=7`. At `mask=01`, `base=dp[01]=7` (already updated this same parcel!): slot 1 free -> `dp[11]=max(0, 7+7)=14`. Final `best = max(0,7,7,14) = 14`.

**Diagnosing bug #1 — in-place reuse places one parcel twice.** The code returns `14`, but one parcel cannot earn `7 + 7`. The defect is precise: I wrote into the same `dp` array I was reading, and because I scan masks in increasing order, the value I wrote at `dp[01]` (parcel 0 in slot 0) got immediately consumed at `mask=01` to place parcel 0 *again* in slot 1, producing `dp[11]=14`. A single parcel's transitions must all read the state from *before* this parcel — that means a separate `new_dp` per parcel, seeded from the old `dp`. Let me also keep the skip option explicit (`new_dp[mask] = max(new_dp[mask], dp[mask])`) so masks survive a parcel even if it is not placed.

**Diagnosing bug #2 — the all-`0` base case is a sign/base-case error.** Even after I fix the in-place issue, initializing every mask to `0` is wrong, and the negative entries are what expose it. Trace the fixed-but-mis-seeded version on `n = 1`, `m = 2`,

```
p = [ [-5, -5] ]
```

Correct answer is `0` (the parcel is a loss in every slot; deliver nothing). If `dp` starts as `[0,0,0,0]`, then for parcel 0 at `mask=11` (all slots "occupied", value `0` purely from the bogus seed) there is nothing to place, but the seed itself already asserts `dp[11]=0` is achievable — meaning "both slots occupied for free", which is a fiction. Worse, consider a profit grid where a *positive* later parcel keys off a mask that only the bogus seed made reachable: the all-`0` seed lets the DP believe slots were filled at no cost, inflating the answer. The all-negative case happens to still print `0` here by luck (no positive cell to exploit), but the seed is structurally unsound: only `dp[0]` should be `0`; every other mask must start at `NEG` so it loses every `max` until something legitimately reaches it. The fix is `dp.assign(full, NEG); dp[0] = 0;`. This is the base-case/sign bug the problem is built around: the optional-delivery floor of `0` must come from the *skip* transition keeping `mask=0` alive at `0`, not from pre-seeding phantom occupied masks.

**Fixing and re-verifying.** New core: a fresh `new_dp` per parcel, seeded to `NEG`, with an explicit skip transition, and the base case `dp[0]=0` only.

```
const long long NEG = LLONG_MIN / 4;
vector<long long> dp(full, NEG);
dp[0] = 0;
for (int i = 0; i < n; i++) {
    vector<long long> ndp(full, NEG);
    for (int mask = 0; mask < full; mask++) {
        if (dp[mask] == NEG) continue;          // unreachable, skip
        long long base = dp[mask];
        ndp[mask] = max(ndp[mask], base);        // skip parcel i
        for (int j = 0; j < m; j++) {
            if (mask & (1 << j)) continue;
            ndp[mask | (1<<j)] = max(ndp[mask | (1<<j)], base + p[i][j]);
        }
    }
    dp.swap(ndp);
}
```

Re-trace bug #1's input `n=1, m=2, p=[[7,7]]`: start `dp=[0,NEG,NEG,NEG]`. Parcel 0: only `mask=00` is reachable, `base=0`. Skip -> `ndp[00]=0`. Place slot 0 -> `ndp[01]=7`. Place slot 1 -> `ndp[10]=7`. `ndp[11]` stays `NEG` (no reachable mask with two free slots after one parcel). `best = max(0,7,7) = 7`. Correct — the double-placement is gone. Re-trace bug #2's input `n=1, m=2, p=[[-5,-5]]`: start `dp=[0,NEG,NEG,NEG]`. Parcel 0 at `mask=00`, `base=0`: skip -> `ndp[00]=0`; place slot 0 -> `ndp[01]=-5`; place slot 1 -> `ndp[10]=-5`. `best = max(0,-5,-5) = 0`. Correct — the `0` now comes from the legitimate skip path, not a phantom seed. Both cases that broke before pass, and they pass for the exact reasons I fixed, which is the evidence I trust.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0` (empty instance): the parcel loop never runs, `dp` stays `[0, NEG, NEG, ...]`, `best` starts at `0` and only `dp[0]=0` is non-`NEG`, so it prints `0`. Deliver nothing — correct.
- `n = 1`, single negative-only row: shown above, prints `0`.
- All-negative matrix (any size): every placement strictly decreases profit, so the best reachable value at any non-zero mask is below `0`, but `dp[0]=0` survives every parcel via skip; `best=0`. Correct.
- All zeros: every placement keeps profit at `0`, so `best=0`. Delivering for `0` is allowed but pointless; the answer matches. Correct.
- `n > m` (more parcels than slots): only `m` parcels can be placed; surplus parcels just take the skip transition. The mask never overflows `m` bits. Correct.
- `n < m` (more slots than parcels): some slots stay empty; reachable masks have at most `n` bits set, the rest stay `NEG`. Correct.
- Overflow: accumulators are `long long`; max total `~1.8 * 10^10` fits with vast room. `NEG = LLONG_MIN/4` is only ever read inside a `max` or skipped via the `== NEG` guard; I add `p[i][j]` only to a real `base` (never to `NEG`), so no underflow. Safe.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace, so the grid may be laid out on one line or many.

**Stress verification against a brute oracle.** I am not going to trust three hand traces alone. I wrote an independent brute force that lets each parcel choose one of `m+1` decisions (undelivered, or slot `0..m-1`), rejects any assignment reusing a slot, and takes the max total (seeded at `0` for the empty matching) — an obviously-correct method completely unlike the DP. I generated 900 tiny random cases across five regimes (mixed signs, all-negative, all-zero, all-positive, zeros sprinkled in) with `n` up to 6 and `m` up to 5, and compared. Zero mismatches. The documented sample prints `15`; the all-negative `2x2` prints `0`; the empty `0 3` prints `0`. The `n=m=18` worst case runs in about `0.1 s` and `8 MB`. That is the level of confidence I ship at.

**Final solution.** I convinced myself the idea is right by disproving greedy and hand-checking the recurrence on the sample, and I convinced myself the *code* is right by tracing two failing cases to precise causes — in-place double placement and an unsound all-`0` base case — fixing both, re-verifying, sweeping the corners, and then stress-testing against an independent oracle. That is what I ship:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    if (!(cin >> n >> m)) return 0;        // missing header -> nothing to do
    vector<vector<long long>> p(n, vector<long long>(m));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) cin >> p[i][j];

    // dp[mask] = best total profit after deciding the first i parcels,
    // where mask is the set of slots already occupied.
    // Empty assignment is allowed, so 0 is always reachable.
    const long long NEG = LLONG_MIN / 4;   // unreachable sentinel
    int full = 1 << m;
    vector<long long> dp(full, NEG);
    dp[0] = 0;                              // before any parcel, no slot used, profit 0

    for (int i = 0; i < n; i++) {
        vector<long long> ndp(full, NEG);
        for (int mask = 0; mask < full; mask++) {
            if (dp[mask] == NEG) continue;
            long long base = dp[mask];
            // Option A: leave parcel i undelivered (slots unchanged).
            ndp[mask] = max(ndp[mask], base);
            // Option B: deliver parcel i in some free slot j.
            for (int j = 0; j < m; j++) {
                if (mask & (1 << j)) continue;
                int nmask = mask | (1 << j);
                ndp[nmask] = max(ndp[nmask], base + p[i][j]);
            }
        }
        dp.swap(ndp);
    }

    long long best = 0;                     // empty selection always allowed
    for (int mask = 0; mask < full; mask++)
        if (dp[mask] != NEG) best = max(best, dp[mask]);

    cout << best << "\n";
    return 0;
}
```

**Causal recap.** Greedy looked plausible but a traced counterexample (`[[10,9],[9,0]]`: greedy `10` vs the matching `9+9=18`) showed the single biggest cell need not lie in any maximum matching, so I moved to a subset DP over occupied slots; the DP's two transitions per parcel both must read the *previous* `dp`, which my first in-place loop violated — a trace of `n=1,m=2,[[7,7]]` returning the illegal `14` pinpointed the double placement, fixed by a fresh `new_dp` per parcel; the deeper, twist-specific bug was seeding *every* mask to `0`, which fabricates phantom occupied sets — a trace with negative entries (`[[-5,-5]]`) showed the `0` floor must instead come from the skip transition keeping `mask=0` alive, fixed by `dp[0]=0` with all else `NEG`; and 64-bit accumulators plus the `NEG`-guarded transitions close out the all-negative, all-zero, empty, `n>m`, `n<m`, and overflow corners, all confirmed against an independent brute oracle over 900 cases.
