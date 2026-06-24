**Problem.** A depot has `n` parcels and `m` slots (`0 <= n <= 18`, `1 <= m <= 18`). Assigning parcel `i` to slot `j` earns net profit `p[i][j]`, which may be positive, zero, or negative. Each slot holds at most one parcel and each parcel goes in at most one slot, and delivery is optional (parcels may stay undelivered, slots may stay empty). Maximize total net profit; since the empty assignment is allowed, the answer is at least `0`. Read `n m` and the `n x m` grid from stdin, print the maximum.

**Why the obvious greedy is wrong.** "Repeatedly lock the largest free positive cell" fails because matching is a global constraint. On `[[10, 9], [9, 0]]` greedy grabs `p[0][0]=10` and stops at `10`, but the matching parcel0->slot1, parcel1->slot0 scores `9 + 9 = 18`. The single biggest entry need not belong to any maximum-weight matching, and committing to it can block a pair two medium cells needed. Greedy is discarded.

**Key idea — subset DP over occupied slots.** Process parcels in order. Let `dp[mask]` = best total profit after deciding the parcels seen so far, with exactly the slots in `mask` occupied. For each parcel, from every reachable `mask`:

- **Skip it:** `ndp[mask] = max(ndp[mask], dp[mask])` (mask unchanged).
- **Place it in a free slot `j`** (bit `j` not in `mask`): `ndp[mask | (1<<j)] = max(..., dp[mask] + p[i][j])`.

Use a fresh `ndp` per parcel so all of one parcel's transitions read the *previous* layer. Base case: `dp[0] = 0`, every other mask `= NEG` (a `LLONG_MIN/4` sentinel). Answer: the max over reachable masks, floored at `0`.

**Correctness.** `dp[mask]` ranges over all partial matchings of the processed parcels onto the slots in `mask`; the two transitions enumerate exactly the choices for the next parcel (undelivered, or into any free slot), so by induction `dp[mask]` is the optimum for that occupied set. The `0` floor is realized because the skip transition keeps `mask=0` alive at value `0` through every parcel, so doing nothing is always an option.

**Pitfalls.**
1. *Wrong base case / sign handling (the core trap).* Seeding **every** mask to `0` fabricates phantom "slots filled for free" states; with negative or zero entries this corrupts the DP. Only `dp[0]=0`; all else must be `NEG` so it loses every `max` until legitimately reached. The `0` floor must come from the skip transition, not from the seed. (A trace on `[[-5,-5]]` exposes this — correct answer is `0`, reached via skipping, not via a phantom full mask.)
2. *In-place layer reuse.* Updating `dp` in place while sweeping masks upward lets one parcel be placed twice (e.g. `n=1,m=2,[[7,7]]` wrongly yields `14` instead of `7`). Build a fresh `ndp` each parcel.
3. *Overflow.* Up to `18` placements of `~10^9` reach `~1.8*10^10`; use `long long`. Never add `p[i][j]` to the `NEG` sentinel (guard reachable masks first), so it cannot underflow.

**Edge cases (all handled by the recurrence + the `0` floor):** `n = 0` -> `0`; a single all-negative row -> `0`; an all-negative matrix -> `0`; all zeros -> `0`; `n > m` (surplus parcels skip); `n < m` (spare slots stay empty).

**Complexity.** `O(n * 2^m * m)` time, `O(2^m)` extra space. At `n = m = 18` that is `~8.5*10^7` ops, about `0.1 s` and `8 MB`.

**Code.**

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
