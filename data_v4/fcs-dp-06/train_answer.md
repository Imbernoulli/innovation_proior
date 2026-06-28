**Problem.** Given `n` non-negative integers `a[0..n-1]` with each `a[i] < 2^20`, count the unordered index pairs `(i, j)`, `i < j`, whose masks are disjoint: `a[i] AND a[j] == 0`. Read `n` and the values from stdin, print the count. Zero is disjoint from everything (including another zero), so pairs of zeros count. `n` up to `10^6`.

**Why the obvious approach is wrong.** The double loop "for every pair test `a[i] AND a[j] == 0`" is correct but `O(n^2)`. At `n = 10^6` that is `~5 * 10^11` AND-tests, hundreds of seconds against a 2 s limit — off by two orders of magnitude. No constant-factor tuning rescues it. The cost must not scale with the number of *pairs*; it should scale with the *value universe*, which has only `2^20` masks regardless of `n`.

**Key idea — sum-over-subsets (SOS) DP over the mask lattice.** Fix index `i`. Its valid partners are the values that share no bit with `a[i]`, i.e. the values that are **submasks** of the complement `comp_i = FULL XOR a[i]` where `FULL = 2^20 - 1`. So if `f[m]` = "how many array values are submasks of `m`", the partner count for `i` is the `O(1)` lookup `f[comp_i]`.

Computing `f[m]` for *all* `2^20` masks is the standard sum-over-subsets (subset-lattice zeta) transform. Start with the value histogram `f[m]` = #values equal to `m`. Then for each bit `b = 0..B-1`, for each mask `m` with bit `b` set, add `f[m without bit b]` into `f[m]`. Bits are independent, so processing one bit per pass merges the "bit `b` matched" and "bit `b` dropped" submask choices; after all `B` passes, `f[m]` = #values that are submasks of `m`. Cost `B * 2^B = 20 * 2^20 ≈ 2.1 * 10^7`, independent of `n`. Total `O(n + B * 2^B)`, near-linear. This is the SOTA shape — you must at least touch the `2^B` lattice, and no asymptotically faster method exists for arbitrary submask-count queries.

**Counting bookkeeping.** Let `ordered = sum over i of f[comp_i]`. Each disjoint unordered pair `{i, j}`, `i != j`, is counted twice (in `f[comp_i]` via `j` and in `f[comp_j]` via `i`). Index `i` counts itself iff `a[i]` is a submask of `comp_i`, i.e. iff `a[i] == 0`; let `z` = number of zeros. So `ordered = 2P + z` where `P` is the desired count, giving `P = (ordered - z) / 2`.

**Pitfalls to get right.**
1. *The self-pair correction order.* Self-hits live *inside* the doubled sum, so subtract `z` *before* halving: `(ordered - z) / 2`, not `ordered/2 - z`. The wrong form double-discounts the zeros (on `[0,0,0]` it gives `1` instead of `3`).
2. *SOS update direction.* Iterate `m` upward and read `f[m ^ (1<<b)]` (bit `b` cleared, a strictly smaller index). That source still has bit `b` = 0 this pass, so it is in the correct partially-transformed state; iterating downward or reading the supermask mixes half-updated values.
3. *Overflow.* The all-zeros input makes the answer `C(n,2) ≈ 5 * 10^11` and `ordered ≈ n^2 = 10^12` at `n = 10^6`; both exceed 32-bit. Use `long long` for `ordered` and the answer. An `int` is a silent wrong-answer.

**Edge cases (all handled by the recurrence + correction):** `n = 0` -> `0`; `n = 1` (zero or not) -> `0`; all values full-mask -> `0`; all zeros -> `C(n,2)`; duplicates handled because `f` starts from the multiplicity histogram.

**Complexity.** `O(n + B * 2^B)` time with `B = 20`; `O(2^B)` extra space (an 8 MB `long long` table), well under 256 MB.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // empty input -> no pairs

    const int B = 20;                      // values are < 2^20
    const int SZ = 1 << B;                 // 1048576 masks
    const int FULL = SZ - 1;               // 0b11..1 (20 ones)

    // f[m] will become "how many array values are submasks of m".
    // Start as the multiplicity histogram of the input values, then
    // run the sum-over-subsets (SOS) transform in place.
    vector<long long> f(SZ, 0);
    vector<int> a(n);
    long long zeros = 0;                   // count of values equal to 0
    for (int i = 0; i < n; i++) {
        int x;
        cin >> x;
        a[i] = x;
        f[x] += 1;
        if (x == 0) zeros++;
    }

    // SOS DP: after bit b, f[m] = #values whose mask is a submask of m
    // when restricted to differing only in bits <= b. After all B bits,
    // f[m] = total #array values that are submasks of m.
    for (int b = 0; b < B; b++) {
        for (int m = 0; m < SZ; m++) {
            if (m & (1 << b)) {
                f[m] += f[m ^ (1 << b)];
            }
        }
    }

    // For index i, the values disjoint from a[i] are exactly the submasks
    // of comp_i = FULL ^ a[i]. f[comp_i] counts them over ALL indices,
    // including i itself iff a[i] == 0 (0 is a submask of everything).
    long long ordered = 0;                  // ordered pairs (i, j), i may equal j
    for (int i = 0; i < n; i++) {
        ordered += f[FULL ^ a[i]];
    }

    // ordered counts: each unordered pair {i,j}, i!=j, disjoint, twice;
    // plus a self-pair (i,i) once for every value 0 (since 0 AND 0 == 0).
    // Remove the self-pairs, then halve to get unordered i<j pairs.
    long long answer = (ordered - zeros) / 2;

    cout << answer << "\n";
    return 0;
}
```
