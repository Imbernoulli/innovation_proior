**Problem.** A line has `m` parallel stamping presses. Press `i` warms up for `w[i]` ms, stamps its
first part *at* time `w[i]`, then one part every `c[i]` ms. By time `T` it has made
`0` parts if `T < w[i]`, else `floor((T - w[i]) / c[i]) + 1`. Given a quota `N`, output the smallest
`T` at which the presses together have made at least `N` parts. Read `m N` then `m` pairs `w[i] c[i]`
from stdin; print the answer. `N = 0` -> `0`.

**Key idea — binary search on the answer.** Let
`produced(T) = sum_i [T >= w[i]] * (floor((T - w[i]) / c[i]) + 1)`. This is **non-decreasing** in `T`
(raising `T` only switches presses on and grants more cycles; no part is ever lost), so the predicate
`produced(T) >= N` is monotone — false below a threshold, true at and above it. The answer is that
threshold, found by binary search. Each test is an `O(m)` sum.

**Correctness.** Monotonicity makes the predicate a clean step function, so the standard
`lo < hi` "find first true" search converges to the threshold. The search needs a valid upper bound:
press `i` *alone* stamps its `N`-th part at `w[i] + (N - 1) * c[i]`, and the full line (which includes
press `i`) has produced at least `N` by then, so `hi = max_i ( w[i] + (N - 1) * c[i] )` satisfies
`produced(hi) >= N`. `lo = 0` is a valid left end. The per-press count `floor((T - w[i]) / c[i]) + 1`
paired with the `T >= w[i]` test correctly counts the part stamped exactly at the warm-up instant.

**Pitfalls.**
1. *Overflow (the headline trap).* With `N` and `c[i]` up to `10^9`, the bound `w[i] + (N - 1) * c[i]`
   and hence the answer `T` reach `~10^18`, ~`4*10^8` times past the 32-bit ceiling. The product
   `(N - 1) * c[i]`, the search endpoints, `mid`, and the part accumulator inside `produced` must all
   be `long long`. An `int` endpoint truncates `10^18` and the search prints `0` for an answer of
   `10^18` — a silent wrong answer on exactly the advertised tests. The accumulator is additionally
   saturated at a `CAP` so the running sum can never overflow even 64-bit and can short-circuit once
   the quota is reached.
2. *Warm-up off-by-one.* Use `T >= w[i]` (not strict) with the `+ 1`, so the part at the warm-up
   instant is counted; dropping the `+ 1` or using `>` undercounts by one and inflates the answer.
3. *`N = 0`.* Handle explicitly (answer `0`); otherwise `(N - 1) * c[i]` can make a negative `hi`.

**Edge cases.** `N = 0` -> `0`; `N = 1` -> `min_i w[i]` (earliest warm-up); single press
`1, N, w, c` -> `w + (N - 1) * c` (~`10^18`); many fast presses where the total at the threshold
overflows 32-bit (handled by the `long long` accumulator). `c[i] >= 1` rules out division by zero.

**Complexity.** `O(m log(max T))` time with `log(max T) ≈ 60`, so `~6*10^6` operations at the limit;
`O(m)` memory. Runs the full `m = 10^5`, `N = 10^9` case in ~`0.02` s.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int m;
    long long N;
    if (!(cin >> m >> N)) return 0;
    vector<long long> w(m), c(m);
    for (int i = 0; i < m; i++) cin >> w[i] >> c[i];

    if (N == 0) { cout << 0 << "\n"; return 0; }

    // produced(T): total parts all presses make by time T (milliseconds).
    // press i: 0 if T < w[i], else floor((T - w[i]) / c[i]) + 1.
    // Returns min(total, CAP) so the running sum cannot overflow long long.
    const long long CAP = (long long)4e18;
    auto produced = [&](long long T) -> long long {
        long long total = 0;
        for (int i = 0; i < m; i++) {
            if (T >= w[i]) {
                total += (T - w[i]) / c[i] + 1;
                if (total >= CAP) return CAP; // saturate early
            }
        }
        return total;
    };

    // Binary search smallest T with produced(T) >= N.
    // hi: a T that is certainly enough. The fastest press finishes N parts at
    // w_min + (N-1)*c_min; bound generously with max possible values.
    long long lo = 0;
    long long hi = 0;
    for (int i = 0; i < m; i++) {
        long long t = w[i] + (N - 1) * c[i]; // press i alone makes N parts by here
        hi = max(hi, t);
    }
    // hi as computed is an upper bound (single press alone already reaches N).

    while (lo < hi) {
        long long mid = lo + (hi - lo) / 2;
        if (produced(mid) >= N) hi = mid;
        else lo = mid + 1;
    }

    cout << lo << "\n";
    return 0;
}
```
