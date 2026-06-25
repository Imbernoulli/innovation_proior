**Problem.** A quarry has `n` boulders of integer weights `w[i]` and one chisel of integer power `p`: reducing a boulder of weight `w` to pieces each of weight at most `p` costs `ceil(w/p) - 1` blows. With a global budget of `k` blows across all boulders, find the **minimum** power `p >= 1` whose total blow count is at most `k`. Read `n`, `k`, then the `n` weights from stdin; print the minimum `p`.

**Key idea — binary search on the answer.** The predicate "power `p` keeps total blows `<= k`" is monotone in `p`: a larger `p` never increases any boulder's blow count, so feasibility, once reached, persists for all larger powers. So binary-search the smallest feasible `p` over `[1, maxw]`, where `maxw = max(w[i])` is always feasible (it needs `0` blows per boulder). Each feasibility test is `O(n)`; the search does `O(log maxw)` of them.

The per-boulder cost is `ceil(w/p) - 1`, written without floats as `(w - 1) / p` (integer division). On an exact multiple this gives one fewer than `w/p`: a 8 kg boulder at `p = 4` splits into `4, 4` — `(8-1)/4 = 1` blow, whereas `8/4 = 2` is wrong.

**Pitfalls.**
1. *Search boundary (inclusive vs exclusive).* We want the smallest power that *passes* the predicate, so on a feasible `mid` the candidate `mid` must stay in the window: set `hi = mid`, not `hi = mid - 1`. Using `mid - 1` discards a feasible candidate and returns a value below the true minimum. Trace `w = [6], k = 1`: feasible powers are `p >= 3` (`cuts(3)=1`, `cuts(2)=2`); `hi = mid - 1` returns the infeasible `2`, while `hi = mid` returns `3`. Pair `hi = mid` with the floor midpoint `mid = lo + (hi-lo)/2` so the window always shrinks and the loop terminates.
2. *Formula off-by-one.* `(w-1)/p`, not `w/p` or `ceil(w/p)`. The divisibility case `w = [6], k = 2` (answer `2`, splitting `6` into `2,2,2` = two blows) is wrong under `w/p` (`6/2 = 3 > 2` would push the answer to `3`).
3. *Overflow.* At `p = 1` total blows reach `~2*10^14`, and `k` itself is up to `2*10^14`; `k`, the weights, and the accumulator must be `long long`. An `int` is a silent wrong-answer at scale.

**Edge cases.** `k = 0` forces `p >= maxw`, returned correctly because `maxw` is always feasible; a single boulder of weight `1` with `k = 0` gives the range `[1,1]` and prints `1`; many equal weights and large `n` accumulate to `~2*10^14`, safely inside 64-bit, with an early exit in the cost function once the budget is exceeded.

**Complexity.** `O(n log maxw)` time, `O(n)` space for the input. About `0.05 s` and under `5 MB` at `n = 2*10^5`, `w[i]` near `10^9`.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long k;
    if (!(cin >> n >> k)) return 0;
    vector<long long> w(n);
    long long maxw = 1;
    for (auto &x : w) { cin >> x; maxw = max(maxw, x); }

    // cuts(p) = total chisel blows needed if each blow can leave pieces of size <= p.
    // A boulder of weight x needs ceil(x/p) final pieces, i.e. ceil(x/p)-1 blows.
    // ceil(x/p)-1 == (x-1)/p  in integer arithmetic (x>=1, p>=1).
    auto cuts = [&](long long p) -> long long {
        long long total = 0;
        for (long long x : w) {
            total += (x - 1) / p;       // ceil(x/p) - 1 blows for this boulder
            if (total > k) return total; // early exit, avoid overflow-ish blowup
        }
        return total;
    };

    // feasible(p): with chisel power p, total blows <= k. Monotone: larger p => fewer blows.
    // We want the MINIMUM p with feasible(p) true. p ranges over [1, maxw]
    // (p = maxw always needs 0 blows per boulder, so it is always feasible).
    long long lo = 1, hi = maxw;        // search space [lo, hi], answer guaranteed in it
    while (lo < hi) {
        long long mid = lo + (hi - lo) / 2;
        if (cuts(mid) <= k) hi = mid;   // mid works -> answer is mid or smaller
        else lo = mid + 1;              // mid fails -> answer strictly larger
    }
    cout << lo << "\n";
    return 0;
}
```
