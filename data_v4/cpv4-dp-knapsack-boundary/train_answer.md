**Problem.** A locker has `K` integer space units (positions `1..K`); fire code reserves the
**last `g` units** as an empty buffer, so only the first `K - g` units are usable. Given `n`
items where item `i` occupies `s[i]` units and is worth `v[i]` (each used at most once), choose
a subset whose total occupied space is at most the usable amount and maximize total value. Read
`n K g` then the `n` pairs from stdin; print the maximum value. The empty subset is legal, so
the answer is at least `0`.

**Why greedy is wrong.** Sorting by value-density `v[i]/s[i]` and grabbing greedily is correct
for *fractional* knapsack but not for 0/1. With `U = 6` and items `(3,5), (3,5), (4,7)`, greedy
takes `(4,7)` first and nothing else fits (total `7`), but the two `(3,5)` items fill `6` for
`10`. Greedy is discarded; this is exact 0/1 knapsack.

**Key idea — the capacity is a boundary difference.** The usable capacity is `U = K - g`,
obtained by counting: positions `1..K` minus the last `g` (positions `K-g+1..K`) leaves exactly
`K - g` usable units — not `K - g + 1`, not `K - g - 1`. Clamp `U = max(K - g, 0)` since `g` may
be `>= K`. Then run the standard 1-D 0/1 knapsack: `dp[c]` = best value with occupied space at
most `c`, for `c = 0..U`. For each item scan `c` downward and apply
`dp[c] = max(dp[c], dp[c - s_i] + v_i)` for `c >= s_i`. The answer is `dp[U]`.

**Correctness.** `dp` starts at `0` (empty subset). Processing items one at a time with `c`
*decreasing* guarantees `dp[c - s_i]` still reflects the table without item `i`, so each item is
counted at most once (ascending order would be unbounded knapsack). Because a larger budget
never hurts, `dp` is non-decreasing in `c`, so `dp[U]` already equals the best value over "space
at most `U`."

**Pitfalls.**
1. *Capacity off-by-one.* `U` is exactly `K - g`. A one-unit slip silently keeps a buffer unit
   or burns a usable one. Derive it by counting positions.
2. *Inclusive boundaries in the DP.* The table needs `U + 1` slots (indices `0..U`); sizing it
   `U` makes `dp[U]` — the printed answer — out of bounds. The scan cutoff must be inclusive
   `c >= s_i`, not `c > s_i`; the exclusive form refuses to place an item into a slot it exactly
   fills, dropping every flush-fitting optimum. A trace of `n=1, U=3, item (3,9)` (correct
   answer `9`) exposes both: the size-`U` table reads out of bounds and the `c > si` cutoff
   never places the item.
3. *Negative capacity.* When `g > K`, `U = K - g < 0`; computing `(size_t)(U + 1)` wraps to a
   huge unsigned value and blows up the allocation. Clamp `U` to `0` *before* sizing the table.
4. *Overflow.* Up to `2000` items each worth `10^9` sum to `~2*10^12`; accumulate in
   `long long`. An `int` is a silent wrong-answer.

**Edge cases.** `g = K` or `g > K` -> `U = 0`, answer `0`. `g = 0` -> plain knapsack over the
full locker. An item with `s_i > U` is skipped (its inner loop never runs). All-too-large items
-> `0`. Single item exactly filling `U` is placed by the inclusive cutoff.

**Complexity.** `O(n * U)` time (`<= 2000 * 2*10^5 = 4*10^8` long-long ops, ~0.2 s measured),
`O(U)` memory (one `dp` row, `~1.6 MB`).

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long K, g;
    if (!(cin >> n >> K >> g)) return 0;
    vector<long long> s(n), v(n);
    for (int i = 0; i < n; i++) cin >> s[i] >> v[i];

    // Usable space: the locker has capacity K, but the last g units must stay
    // empty (fire-safety buffer). Positions are 1..K; reserving the last g of
    // them leaves usable positions 1..(K-g), i.e. U = K - g usable units.
    // If K - g < 0 there is no usable space at all -> treat U = 0.
    long long U = K - g;
    if (U < 0) U = 0;               // no usable space at all

    // 0/1 knapsack over capacity 0..U inclusive.
    // dp[c] = best value achievable using total occupied space exactly <= c.
    vector<long long> dp((size_t)U + 1, 0);
    for (int i = 0; i < n; i++) {
        long long si = s[i], vi = v[i];
        if (si > U) continue;                 // item cannot fit at all
        for (long long c = U; c >= si; c--) { // 0/1: descending
            long long cand = dp[c - si] + vi;
            if (cand > dp[c]) dp[c] = cand;
        }
    }

    cout << dp[U] << "\n";
    return 0;
}
```
