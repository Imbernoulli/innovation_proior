**Reading the problem and pinning the contract.** I have `m` time slots `0..m-1` and `n` offered bands. Booking band `i` makes it play a contiguous block of `d_i` slots starting at `s_i`, so it occupies `s_i, s_i+1, ..., s_i+d_i-1` — I want to be very deliberate that this is the **half-open** interval `[s_i, s_i+d_i)`, because every bug in this problem is going to live in that `-1`. A line-up is a subset of bands that (a) each fit entirely in `[0, m)` and (b) pairwise share no slot; I maximize the total profit and the empty line-up earns `0`, so the answer never drops below `0`. Input is `m n` then `n` triples `s d p`; output is one integer.

Before any algorithm I fix the scale, because it decides the data types and the whole shape of the solution. `m <= 16`, so the set of occupied slots is a subset of a 16-element universe — `2^16 = 65536` masks, which screams bitmask DP. `n <= 2*10^5` and `p_i <= 10^9`, so the total profit can in principle approach `m * 10^9 = 1.6*10^10` (at most `m` non-overlapping bands of length 1). That already exceeds the 32-bit ceiling of about `2.1*10^9`, so the profit accumulators must be `long long`. Reading `s, d` as `long long` too is cheap insurance: I am going to compute `s + d` and compare it to `m`, and if I ever let `s + d` overflow a small int with adversarial-but-in-range values I would mis-decide fit. So: 64-bit for profit and for the fit arithmetic, 32-bit `int` only for masks (16 bits, safe).

**Laying out the candidate approaches.** Two routes, and I want the one I can *prove*.

- *Greedy by profit.* Sort bands descending by `p`, book each if its slots are still free. `O(n log n)`, three lines. But slot packing is a global constraint: a single fat band can block two thin bands that jointly beat it. I distrust this and will try to break it before trusting it.
- *Bitmask DP over the occupied-slot set.* A mask says which slots are already used. The clean way to avoid counting the same line-up many times is to impose an order: always work on the **lowest free slot** of the current mask, and from there either leave that slot empty forever or start exactly one band whose first slot is that slot. Each line-up is then built in a unique left-to-right way. This is the route the problem is shaped for; the only real risk is boundary transcription.

**Stress-testing greedy before committing.** "Greedy feels fine" is how wrong code ships. Let me attack it. `m = 4`. Band A occupies slots `{0,1,2,3}` (`s=0, d=4`) with profit `10`. Band B occupies `{0,1}` (`s=0, d=2`) profit `6`, band C occupies `{2,3}` (`s=2, d=2`) profit `6`. Greedy sorts by profit: A (10) first, books it, fills all slots; B and C now collide and are rejected. Greedy total `10`. But B + C are disjoint and give `12 > 10`. Greedy is wrong, and I can see exactly why — grabbing the biggest single band can block strictly more than it earns. Greedy is out; I commit to the bitmask DP.

**Deriving the DP and being explicit about every boundary.** I define `dp[mask]` = the maximum total profit achievable by a set of pairwise slot-disjoint bands whose union of occupied slots is a **subset** of `mask`. Base: `dp[0] = 0` (book nothing). I fill masks in increasing numeric order. The transition uses the lowest-free-slot rule:

- For a given `mask`, let `low` be its lowest slot that is **not** set. (If every slot is set, there is nothing to do.)
- *Leave `low` empty.* Then `dp[mask | (1<<low)] = max(that, dp[mask])`. This just advances the "frontier" past `low` without booking anything there.
- *Start a band at `low`.* For every band whose **lowest occupied slot is exactly `low`** and whose mask does not collide with `mask`, set `dp[mask | bandmask] = max(that, dp[mask] + profit)`. Requiring the band's lowest slot to equal `low` is what makes each line-up reachable by exactly one sequence of decisions, so I never double-count and I never miss one.

The answer is `max over all masks of dp[mask]` (equivalently `dp` of the full mask, but taking the max over all masks is robust and includes the empty line-up at `dp[0] = 0`).

Now the three boundary facts I must nail, each one `+/-1` from a wrong variant:

1. **Which slots a band occupies.** `[s, s+d)` = the loop `for k in s .. s+d-1`. Writing `k <= s+d` (inclusive) would steal an extra slot; writing `k < s+d-1` would drop one.
2. **When a band fits.** Last occupied slot is `s+d-1`, and it must be `<= m-1`, i.e. `s + d <= m`. The plausible-but-wrong variants are `s + d - 1 <= m` (lets a band overrun by one) or `s + d < m` (forbids a band that legally ends on the last slot).
3. **Where a band may start in the DP.** A band may only be *started* on the slot equal to its lowest occupied slot, which is `s`. If I let a band be started on a higher free slot I'd both double-count and, worse, place it where its lower slots collide with already-used ones.

**Sanity-checking the derivation on the worked sample.** `m = 5`, bands B0 `(0,2,6)` -> slots `{0,1}`; B1 `(2,3,9)` -> `{2,3,4}`; B2 `(1,2,5)` -> `{1,2}`; B3 `(3,2,7)` -> `{3,4}`. By hand: B0 `{0,1}` and B1 `{2,3,4}` are disjoint and cover everything for `6+9=15`. Any other disjoint pair? B0 `{0,1}` + B3 `{3,4}` = `6+7=13` (slot 2 idle). B2 `{1,2}` + B3 `{3,4}` = `5+7=12`. So `15` is optimal, and that is the documented answer. Good — the model of the problem matches the intended answer.

**First implementation — and a trace, because clean math transcribes dirty.** Here is my first cut of the band-reading loop, where the danger is the two endpoints:

```
// FIRST TRY (buggy)
if (d <= 0 || s < 0 || s + d - 1 > m) continue;  // "fits"
int mask = 0;
for (long long k = s; k < s + d; k++) mask |= (1 << (int)k);
```

I trace the smallest input that can expose a fit-boundary bug: `m = 4`, bands `(0,2,5)` -> `{0,1}` profit 5, `(2,2,8)` -> `{2,3}` profit 8, and the trap band `(3,2,100)` which *should* occupy slots `3,4`. Slot `4` does not exist in `[0,4)`, so this band must be rejected. Run the fit test on the trap band: `s + d - 1 = 3 + 2 - 1 = 4`, and `4 > m = 4` is **false**, so the buggy test *accepts* it. Then the occupancy loop runs `k = 3, 4` and does `mask |= (1<<3)` and `mask |= (1<<4)`. Now bit 4 is set — a slot that is outside the strip entirely. In the DP, this band's mask `0b11000` has lowest slot 3, so it would be startable at slot 3, would not collide with `{0,1}`... and the DP would happily book it for profit `100`, producing `5 + 100 = 105` (or `100` alone), whereas the correct answer is `5 + 8 = 13`.

**Diagnosing the first bug.** The defect is precise and is exactly the inclusive/exclusive boundary the problem is about. A band's slots are the half-open range `[s, s+d)`; its **last** slot is `s+d-1`; it fits iff that last slot is a real slot, `s+d-1 <= m-1`, which simplifies to `s + d <= m`. My condition `s + d - 1 > m` is off by one: it only rejects when the last slot exceeds `m` (an index of `m+1` or more), so it permits a band whose last slot is exactly `m` — one past the end. The fix is to reject when `s + d > m`. Equivalently: compare the *count* form (`s + d <= m`) rather than a mangled index form. I also note a latent second bug riding along: even with the fit test fixed, if I had *not* fixed it, the occupancy loop set bit `m`, polluting masks with an out-of-range bit; tightening the fit test to `s + d <= m` guarantees `k` never reaches `m`, so no stray high bit is ever set. One fix closes both holes.

**Fixing and re-verifying the fit boundary.** New band-reading loop:

```
if (d <= 0 || s < 0 || s + d > m) continue;   // fits iff last slot s+d-1 <= m-1
int mask = 0;
for (long long k = s; k < s + d; k++) mask |= (1 << (int)k);   // [s, s+d)
```

Re-trace the trap input `m = 4`, band `(3,2,100)`: `s + d = 3 + 2 = 5`, and `5 > 4` is **true**, so the band is correctly dropped. The remaining bands `(0,2,5) -> {0,1}` and `(2,2,8) -> {2,3}` give `5 + 8 = 13`. And the boundary band that should be *allowed* — say `(2,2,X)` with `m = 4`: `s + d = 4`, `4 > 4` is false, so it is kept, occupying slots `2,3` (the loop runs `k = 2,3`, never `4`). Both directions of the boundary are now correct: a band ending exactly on the last slot is allowed, a band ending one past it is rejected. The case that broke now passes, and it passes for the reason I fixed.

**Second implementation — the DP loop, and a second trace, because the start-slot boundary is just as sharp.** My first DP body looked like this:

```
// FIRST TRY (buggy): try EVERY band that does not collide, regardless of its lowest slot
for (int mask = 0; mask < full; mask++) {
    long long cur = dp[mask];
    for (int i = 0; i < realBands; i++) {
        int bm = bandMask[i];
        if (bm & mask) continue;          // no slot collision
        int nm = mask | bm;
        dp[nm] = max(dp[nm], cur + profit[i]);   // book band i on top of mask
    }
}
```

This *looks* fine — it only books a band when its slots are free, which can never produce an overlapping line-up, so it is not wrong in the "illegal output" sense. But I distrust it as a *maximization* because I dropped the lowest-free-slot ordering, and dropping structure in a subset DP usually means I am no longer guaranteeing I explore the right masks in the right order. Let me trace a tiny case to be sure it actually maximizes. `m = 3`. Bands: P `{0}` profit 4, Q `{1}` profit 4, R `{2}` profit 4. The optimum is obviously `P+Q+R = 12`. Process masks in order. `mask=000, cur=dp[000]=0`: booking P -> `dp[001]=4`; Q -> `dp[010]=4`; R -> `dp[100]=4`. `mask=001, cur=dp[001]=4`: Q -> `dp[011]=8`; R -> `dp[101]=8`. `mask=010, cur=dp[010]=4`: P -> `dp[011]=max(8,8)=8`; R -> `dp[110]=8`. `mask=011, cur=dp[011]=8`: R -> `dp[111]=12`. Final `12`. Here it happened to work.

So this version does not obviously break on *that* case — but I am suspicious precisely because it "works by luck of iteration order," and that is exactly the kind of thing that bites on a boundary. Let me push harder with a case where a band's lowest slot is **not** the lowest free slot, which is the configuration the start-slot boundary is about. `m = 3`. Band X occupies `{1,2}` (`s=1, d=2`) profit 10; band Y occupies `{0}` profit 3. The optimum is `X + Y = 13`. Trace: `mask=000, cur=0`: X -> `dp[110]=10`; Y -> `dp[001]=3`. `mask=001, cur=3`: X does not collide (`110 & 001 = 0`) -> `dp[111]=max(?,13)=13`; `mask=010, cur=dp[010]=0` (never set, stays 0): X collides? `110 & 010 = 010 != 0`, skip. ... eventually `dp[111]=13`. Still right.

The "try every non-colliding band" DP is in fact *correct for maximization* — because any independent set of bands can be added in any order and the subset-DP over masks reaches every reachable union. So this is not a correctness bug. But it has a real **complexity** problem I now see: it scans all `n` bands for every one of the `2^m` masks regardless of whether they could be relevant, and worse, it re-derives the same union many times along many orderings, which at `n = 2*10^5` and `2^16` masks is `1.3*10^10` band-checks — too slow for 2 seconds. The lowest-free-slot ordering is what fixes *both* the double-work and lets me bucket bands by their start slot so each mask only scans the bands that can start at its lowest free slot. So I keep the ordered version for performance and group bands by `s`.

But the ordered version has its *own* boundary I must verify: "a band may be started at `low` iff its lowest occupied slot is `low`." If I instead allowed starting any non-colliding band at `low` while *also* advancing only through `low`, I could skip past a slot a band needs. Concretely, suppose I bucket band X `{1,2}` under `byLow[1]` (its lowest slot is 1) — correct. If I had bucketed it under `byLow[0]` by an off-by-one in computing its lowest slot (e.g. taking `s-1` or the wrong bit), then at `mask=000` with `low=0` I would try to "start X at slot 0," union to `{1,2}`, leaving slot 0 free, and later try to start Y `{0}` — fine here, but in a case where X's lowest slot is mis-recorded the frontier ordering breaks and some optimal mask is never produced. So I record the start bucket as exactly `s` (the true lowest slot, since the slots are the *contiguous* range `[s, s+d)` and the lowest is `s`), and in the loop I only ever try bands from `byLow[low]`. I also still guard `if (bm & mask) continue;` so a band whose higher slots collide is rejected. With `low` always being the lowest free slot and the band's lowest slot equal to `low`, the band's lower slots can never be below `low` (none exist) and can never already be occupied (its lowest is `low`, which is free by definition), so the only collision possible is on its *upper* slots, which the guard catches.

**Re-verifying the ordered DP on the start-slot case.** Re-run `m = 3`, X `{1,2}` p10 in `byLow[1]`, Y `{0}` p3 in `byLow[0]`. `mask=000`: lowest free slot `low=0`. Leave-empty -> `dp[001]=0`. Bands in `byLow[0]`: Y `{0}`, no collision -> `dp[001]=max(0,3)=3`. `mask=001`: `low=1`. Leave-empty -> `dp[011]=3`. Bands in `byLow[1]`: X `{1,2}`=`110`, `110 & 001 = 0` -> `dp[111]=max(?,3+10)=13`. `mask=010` has `dp=0` (unreached as a useful state), `low=0`, `byLow[0]` Y `{0}` no collide -> `dp[011]=max(3,3)`... and so on. Final `dp[111]=13`. Correct, and now reached through the unique lowest-slot path, so no double counting and it is fast.

**Edge cases, deliberately, because this is where boundary code dies.**
- `n = 0`: the read loop does nothing, all `dp[mask]=0`, answer `max = 0`. Empty line-up — correct.
- `m = 1`, single band `(0,1,7)`: fits since `0 + 1 = 1 <= 1`; mask `{0}` in `byLow[0]`. `mask=0, low=0`: leave-empty -> `dp[1]=0`; book -> `dp[1]=7`. Answer `7`. Correct. With `m=1` and a band `(0,2,…)`: `0+2=2 > 1`, dropped — correct, it cannot fit one slot.
- Boundary band that ends exactly on the last slot, `m=5`, `(3,2,…)` -> slots `{3,4}`, `3+2=5 <= 5`, kept. Allowed — correct.
- Overrun-by-one band `(s,d)` with `s+d = m+1`, e.g. `m=4`, `(3,2,100)`: dropped, answer ignores its 100 — correct (this is the trap).
- `s = m` (start past the strip): `s + d > m` always (since `d >= 1`), dropped. Correct — there is no slot `m`.
- Overflow: profits accumulate in `long long`; the max realistic sum `~1.6*10^10` fits with vast room. The fit arithmetic `s + d` is computed in `long long`, so even at the input maxima it cannot wrap. Masks use bits `0..m-1 <= 15`, never overflowing `int`.
- Output: exactly one integer and a newline; `cin >>` skips arbitrary whitespace, so input formatting is irrelevant.

**Final solution.** I disproved greedy with a concrete block-more-than-you-earn instance, fixed the *fit* boundary (`s + d <= m`, not `s + d - 1 <= m`) by tracing the `m=4 / (3,2,100)` trap that wrongly produced `105`, and fixed the *start-slot / performance* structure by tracing the `{1,2}` vs `{0}` case and grouping bands by their true lowest slot `s`. This is what I ship — one self-contained file:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int m, n;
    if (!(cin >> m >> n)) return 0;          // m = number of slots, n = number of bands

    // For each band, build the bitmask of slots it occupies. A booked band starting at
    // slot s with duration d covers the HALF-OPEN interval [s, s+d): slots
    // s, s+1, ..., s+d-1. It is bookable only if it fits entirely inside [0, m), which
    // means s + d <= m (equivalently its last slot s+d-1 <= m-1). Bands that run past
    // the last slot are simply unbookable and are dropped.
    // Group usable bands by their lowest occupied slot so the DP only scans relevant ones.
    vector<vector<pair<int,long long>>> byLow(m); // byLow[s] = {(mask, profit), ...}
    for (int i = 0; i < n; i++) {
        long long s, d, p;
        cin >> s >> d >> p;
        if (d <= 0 || s < 0 || s + d > m) continue;   // does not fit in [0, m)
        int mask = 0;
        for (long long k = s; k < s + d; k++) mask |= (1 << (int)k);   // [s, s+d)
        long long prof = (p > 0 ? p : 0); // a non-positive band is never worth booking
        byLow[(int)s].push_back({mask, prof});
    }

    int full = 1 << m;
    // dp[mask] = max total profit using pairwise slot-disjoint bands whose union of
    // occupied slots is a SUBSET of mask. dp[0] = 0 (book nothing). We fill masks in
    // increasing order; from each mask we look at its lowest FREE slot `low` and either
    // leave it empty, or start one band there (a band whose lowest slot is exactly low).
    vector<long long> dp(full, 0);
    for (int mask = 0; mask < full; mask++) {
        long long cur = dp[mask];
        int low = -1;
        for (int b = 0; b < m; b++) if (!(mask & (1 << b))) { low = b; break; }
        if (low == -1) continue;                       // every slot occupied
        int leaveEmpty = mask | (1 << low);            // option: slot `low` stays empty
        if (cur > dp[leaveEmpty]) dp[leaveEmpty] = cur;
        for (auto &pr : byLow[low]) {                  // bands whose first slot is `low`
            int bm = pr.first;
            if (bm & mask) continue;                   // overlaps an occupied slot
            int nm = mask | bm;
            long long val = cur + pr.second;
            if (val > dp[nm]) dp[nm] = val;
        }
    }

    long long ans = 0;
    for (int mask = 0; mask < full; mask++) ans = max(ans, dp[mask]);
    cout << ans << "\n";
    return 0;
}
```

**Causal recap.** Greedy looked right but a single traced instance (one fat `{0,1,2,3}`=10 band blocking `{0,1}`=6 + `{2,3}`=6 = 12) showed a local profit-grab blocks more than it earns, so I committed to a bitmask DP over occupied slots. The whole difficulty was boundary arithmetic: a band occupies the half-open `[s, s+d)`, so it fits iff `s + d <= m` — my first fit test `s + d - 1 > m` was off by one and a traced trap band `(3,2,100)` with `m=4` slipped through to give `105` instead of `13`, fixed by comparing counts `s + d <= m`; and the DP must start a band only on its lowest slot `s` under a lowest-free-slot ordering, which a second trace (`{1,2}`=10 vs `{0}`=3) confirmed and which also makes the per-mask scan cheap by bucketing bands under `byLow[s]`; `long long` profit accumulators and 64-bit fit arithmetic close the overflow corner, and `max(dp[mask])` (including `dp[0]=0`) closes the empty-line-up corner.
