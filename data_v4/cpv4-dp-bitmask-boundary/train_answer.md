**Problem.** A main stage has `m` consecutive slots `0..m-1` (`m <= 16`). Each of `n` offered bands, if booked, plays the half-open slot interval `[s_i, s_i+d_i)` for profit `p_i`. A line-up is a subset of bands that each fit inside `[0, m)` (i.e. `s_i + d_i <= m`) and pairwise share no slot. Maximize total profit; the empty line-up earns `0`. Read `m n` then `n` triples `s d p` from stdin, print the maximum.

**Why the obvious greedy is wrong.** "Sort bands by profit and book each one whose slots are still free" fails because slot packing is global. With `m = 4`, a fat band on `{0,1,2,3}` worth `10` blocks two thin bands on `{0,1}` and `{2,3}` worth `6` each; greedy takes `10`, but the pair gives `12`. A single high-profit band can block strictly more value than it earns, so greedy is discarded.

**Key idea — bitmask DP over occupied slots.** The occupied-slot set is a subset of a 16-element universe, so let `dp[mask]` = max total profit of a set of pairwise slot-disjoint bands whose union of occupied slots is a subset of `mask`, with `dp[0] = 0`. To build each line-up exactly once, process slots left to right: from `mask`, take its lowest **free** slot `low` and either

- leave `low` empty: `dp[mask | (1<<low)] = max(., dp[mask])`; or
- start a band whose lowest occupied slot is exactly `low` and that does not collide with `mask`: `dp[mask | bm] = max(., dp[mask] + profit)`.

Bucketing bands by their start slot `s` (their lowest occupied slot) means each mask only scans the bands that can start at its lowest free slot. The answer is `max` of `dp[mask]` over all masks (which includes `dp[0] = 0`).

**Correctness.** Every line-up is reached by a unique decision sequence (always act on the current lowest free slot), so no double counting and no omissions. A band started at `low` has no slot below `low` and its lowest slot `low` is free by construction, so the only possible collision is on its upper slots, which the `bm & mask` guard rejects — every produced mask corresponds to a genuinely disjoint set of fitting bands, and every disjoint fitting set is producible.

**Pitfalls.**
1. *Inclusive/exclusive fit boundary (the trap).* A band's slots are `[s, s+d)`; its last slot is `s+d-1`, so it fits iff `s+d-1 <= m-1`, i.e. `s + d <= m`. The off-by-one variant `s + d - 1 > m` (or `s + d - 1 <= m`) wrongly admits a band whose last slot is `m` — one past the strip. On `m=4` with a band `(3,2,100)` (slots `3,4`, slot `4` does not exist), the buggy test books it for `105` instead of the correct `13`. Compare counts (`s + d <= m`), not a mangled index.
2. *Occupancy loop.* Fill slots `for k in [s, s+d)` — `k <= s+d` steals an extra slot, `k < s+d-1` drops one. With the fit test tightened to `s + d <= m`, `k` never reaches `m`, so no out-of-range bit is ever set.
3. *Start-slot ordering.* A band may be *started* only on its lowest slot `s`. Allowing it to start on a higher free slot breaks the unique-path ordering and bloats the per-mask scan to `O(2^m * n)`.
4. *Overflow.* Up to `m` length-1 bands at `p_i = 10^9` sum to `~1.6*10^10` > 32 bits; use `long long` for profits, and compute `s + d` in 64-bit so the fit test cannot wrap.

**Edge cases.** `n = 0` -> `0` (no band booked); `m = 1` with `(0,1,p)` -> `p`; a band ending exactly on the last slot (`s + d = m`) is allowed; a band overrunning by one (`s + d = m + 1`) is dropped; `s = m` is always dropped (no slot `m`). All handled by `s + d <= m` plus `max(dp[mask])`.

**Complexity.** `O(2^m * (m + B))` where `B` is the number of bands bucketed at a given slot; reading is `O(n)`. With `m = 16` and `n = 2*10^5` this runs in a few milliseconds and a few MB.

**Code.**

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
