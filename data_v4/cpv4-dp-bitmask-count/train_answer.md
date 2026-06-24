**Problem.** There are `n` employees (`n <= 18`) and `m <= 2*10^5` candidate squads, each a bitmask
over the employees. A *roster* is a set of candidate squads that partitions the whole staff (pairwise
disjoint, union = everyone). Rosters using the same set of squads are identical (order of squads does
not matter). Count the valid rosters modulo `1_000_000_007`. Read `n m` then the `m` masks from stdin;
print the count. Candidate masks may be empty (`0`, ignore), duplicated (collapse to one), or carry
stray high bits (mask them off).

**Key idea — bitmask "next block" DP with a canonical block order.** Let `dp[mask]` be the number of
valid partitions of the employee set `mask`, modulo `p`. Base case `dp[0] = 1` (the empty set has one
partition: use no squads). To fill `dp[mask]`, decide which squad covers the **lowest-numbered**
uncovered employee `low = mask & (-mask)` first. That squad is some allowed submask `sub` of `mask`
that contains `low`; the rest is a partition of `mask ^ sub`:

  `dp[mask] = sum over allowed sub ⊆ mask with (sub & low) ≠ 0 of dp[mask ^ sub]`.

Answer: `dp[full]` where `full = (1<<n) - 1`.

**Why it counts each roster exactly once.** Always committing the block that owns the lowest uncovered
employee forces a unique generation order for every unordered partition: the remainder `mask ^ sub`
has a strictly larger lowest employee, so the recursion never revisits the choice for `low`. Each
partition therefore corresponds to exactly one chain of choices. (Independent check: when every
non-empty subset is allowed, `dp[full]` equals the Bell number `B(n)`; `n=18` gives `76801385`, which
matches `B(18) mod p`.)

**Pitfalls.**
1. *Double-counting by block order.* If you let *any* allowed submask be the next block (no `low`
   guard), a `k`-block roster is counted `k!` times. Trace `n=2`, allowed `{0},{1},{0,1}`: without the
   guard you get `3`; with it you get the correct `2`. The guard `if (!(sub & low)) continue;` is
   mandatory.
2. *Duplicates and the empty squad.* A duplicated candidate mask must not double the count, and an
   empty squad (`0`) must never be a block (it self-loops: `dp[mask] += dp[mask ^ 0] = dp[mask]`).
   Fold candidates into an idempotent boolean `allowed[mask]` table (marking twice is harmless) and
   `continue` on `x == 0`. The submask loop `sub = mask; sub; sub = (sub-1)&mask` never yields `0`, so
   the empty block can never sneak in.
3. *Out-of-range bits.* `x &= full` before the empty test so stray high bits are dropped and never index
   `allowed` out of bounds.
4. *Overflow / modulus.* Counts explode (Bell numbers); use `long long` and reduce `% MOD` after each
   state and at output. The inner sum stays well under `9.2e18`.

**Edge cases.** `n = 0` -> `dp[0] = 1` (loop never runs); no usable squads / impossible coverage -> `0`;
duplicated/empty masks handled by the boolean table; large counts handled by the modulus.

**Complexity.** `O(3^n)` time (`sum_mask 2^popcount(mask)`), `O(2^n)` memory. At `n=18` that is
`~3.9e8` ops, about `0.4 s` and `~19 MB`.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

const long long MOD = 1000000007LL;

int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;

    int full = (n > 0) ? ((1 << n) - 1) : 0;

    // allowed[mask] = 1 iff `mask` is one of the candidate squads, AFTER
    // discarding empty squads, masking off out-of-range bits, and de-duplicating
    // (a repeated mask is still just one allowed squad).
    vector<char> allowed(full + 1, 0);
    for (int j = 0; j < m; j++) {
        int x;
        if (scanf("%d", &x) != 1) x = 0;
        x &= full;             // drop any bit >= n
        if (x == 0) continue;  // empty squad is not a real squad
        allowed[x] = 1;        // marking twice is harmless -> dedup for free
    }

    // dp[mask] = number of ways to partition the employee set `mask` into a
    // collection of candidate squads (each used at most once, order irrelevant),
    // modulo MOD. dp[0] = 1: the empty set has exactly one partition (use no
    // squads).
    //
    // Canonical order to avoid double-counting: the squad that owns the LOWEST
    // remaining employee is decided first. So we only enumerate squads `sub`
    // that are submasks of `mask` AND contain mask's lowest set bit. Each
    // unordered partition is then generated exactly once.
    vector<long long> dp(full + 1, 0);
    dp[0] = 1;
    for (int mask = 1; mask <= full; mask++) {
        int low = mask & (-mask);          // lowest set bit of mask
        long long ways = 0;
        for (int sub = mask; sub; sub = (sub - 1) & mask) {
            if (!(sub & low)) continue;    // squad must own the lowest employee
            if (allowed[sub]) ways += dp[mask ^ sub];
        }
        dp[mask] = ways % MOD;
    }

    printf("%lld\n", dp[full] % MOD);
    return 0;
}
```
