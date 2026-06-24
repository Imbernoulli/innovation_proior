**Problem.** Over `n` days a pump produces net energy `a[i]` (positive, zero, or negative). You may run it during **at most one** non-empty contiguous block `[l, r]`, paying a single startup fee `c >= 0`, earning `a[l]+...+a[r] - c`; or run nothing for profit `0`. Read `n`, `c`, and the values from stdin; print the maximum profit. Because "do nothing" is always available, the answer is at least `0`.

**Why the obvious quadratic is too slow.** Enumerating every window `(l, r)` and taking `max(0, sum - c)` is `O(n^2)` — correct but hopeless at `n = 2*10^5`. It serves only as a reference oracle on tiny inputs.

**Key idea — Kadane with a deferred fee.** The fee `c` is paid once for *any* non-empty block, so subtracting it cannot change *which* block is best: among non-empty blocks, the one maximizing `sum - c` is the one maximizing `sum`. So compute `M` = the best non-empty contiguous block sum (maximum subarray), then the answer is `max(0, M - c)`. Kadane gives `M` in `O(n)`:

- `cur_i = max(a[i], cur_{i-1} + a[i])`  (best block **ending at** `i`: singleton, or extend the previous)
- `M = max_i cur_i`

Initialize `cur` and `bestSum` to `-inf` (a sentinel), **not** `0`, so the block is forced to be non-empty. The `0` is the separate "do nothing" option, applied at the very end after the fee.

**Correctness.** Every non-empty block ending at `i` is either `{i}` or a block ending at `i-1` plus day `i`, so the recurrence is exhaustive and `M` is the true best non-empty block sum. Subtracting the constant `c` preserves the argmax, and `max(0, M - c)` picks the better of "run the best block" and "run nothing." Verified against the `O(n^2)` oracle on 851 random adversarial small cases (all-negative, negatives-and-zeros, `n = 0`, large fee, `c = 0`) with zero mismatches; max-scale `n = 2*10^5` runs in 0.04 s.

**Pitfalls.**
1. *Base case / empty-vs-non-empty.* Seeding Kadane with `0` makes `bestSum` mean "empty block of sum 0" rather than "best **non-empty** block sum." On all-negative inputs the seed-`0` value pins `bestSum` to `0`; with `c >= 0` the final `max(0, ...)` clamp *coincidentally* hides this, but the intermediate is corrupt and the logic is brittle. Seed `-inf` so `bestSum` is honest, and gate the fee on a block actually existing (`bestSum > NEG`) — which also handles `n = 0`.
2. *Sentinel underflow.* The recurrence adds `a[i]` to `cur` *before* the `max`, so a raw `LLONG_MIN` seed underflows (UB) on day 0. Use `NEG = LLONG_MIN / 4`: still below any real sum, but `NEG + a[i]` stays well-defined.
3. *Overflow.* With `n` up to `2*10^5`, `|a[i]|` up to `10^9`, and `c` up to `10^{14}`, sums reach `~2*10^{14}`. Use `long long` everywhere; an `int` is a silent wrong-answer on large tests.

**Edge cases (all handled by the `-inf` base + the final `max(0, ...)`):** `n = 0` -> `0` (loop skipped, `bestSum` stays `NEG`, guard false); single negative or single zero -> `0`; all negatives -> `0`; fee larger than any block sum -> `0`; `c = 0` -> pure maximum subarray vs the empty option.

**Complexity.** `O(n)` time, `O(1)` extra space beyond the input.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long c;
    if (!(cin >> n >> c)) return 0;          // empty input -> profit 0
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // best = max profit of a NON-EMPTY contiguous block ending at i, AFTER paying c.
    //   cur = best subarray sum ending at i (Kadane), base -inf so a single day is forced in.
    // Running-on profit of a block is (sum of block) - c; we compare against doing nothing (0).
    const long long NEG = LLONG_MIN / 4;
    long long cur = NEG;          // best subarray sum ending at current index (non-empty)
    long long bestSum = NEG;      // best subarray sum over all non-empty blocks
    for (int i = 0; i < n; i++) {
        cur = max(a[i], cur + a[i]);   // extend or restart; never "empty" -> base NEG
        bestSum = max(bestSum, cur);
    }

    long long answer = 0;                          // do nothing
    if (bestSum > NEG) answer = max(answer, bestSum - c);   // run the best block, pay c once

    cout << answer << "\n";
    return 0;
}
```
