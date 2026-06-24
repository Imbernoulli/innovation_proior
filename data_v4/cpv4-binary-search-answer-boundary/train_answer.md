**Problem.** `n` flower beds in a row; bed `i` blooms on day `b[i]` and stays bloomed. A bouquet is `k`
consecutive simultaneously-bloomed beds, each bed used once; a maximal bloomed run of length `L`
yields `floor(L/k)` bouquets. Find the earliest day on which at least `m` bouquets can be assembled,
or `-1` if impossible. Read `n m k` and the `n` bloom days from stdin; print the day or `-1`.

**Key idea — binary search on the answer day.** Feasibility is monotone in the day: no bed ever
un-blooms, so if `m` bouquets are makeable on day `t` they are makeable on every later day. The
answer is therefore the lower-bound transition point of `feasible(t)`. Two ingredients:

- *Feasibility test, `O(n)`.* For a fixed day `t`, scan left to right keeping the length `run` of the
  current bloomed streak; each time `run` reaches `k`, count one bouquet and reset `run = 0`; a
  non-bloomed bed resets `run = 0`. This sums `floor(L/k)` over maximal runs. Stop early once
  `bouquets >= m`.
- *Search range.* Feasibility only changes on days when some bed newly blooms, so the earliest
  feasible day is one of the `b[i]`. Search the closed interval `[min b, max b]`.

**Correctness.** Monotonicity makes the lower-bound binary search valid. The answer exists iff
`m*k <= n` (on day `max b` the whole row is one run of length `n`, giving `floor(n/k) >= m` exactly
when `n >= m*k`); otherwise print `-1`. Under `m*k <= n`, `feasible(max b)` is true, so the loop
`while (L<R) { mid = L+(R-L)/2; if (feasible(mid)) R=mid; else L=mid+1; }` converges to the smallest
`t` with `feasible(t)` — each step strictly shrinks `R-L`, and on exit `L` is that earliest day.

**Pitfalls.**
1. *Inclusive-vs-exclusive day boundary.* A bed blooms *on* day `b[i]`, so "bloomed by day `t`" is
   `b[i] <= t`, **not** `b[i] < t`. Using `<` makes every bed usable one day too late and inflates the
   answer (a trace of the sample yields `5` instead of `4` because the bed with bloom day `4` is
   wrongly excluded on day `4`).
2. *Binary-search bound off-by-one.* Keep the candidate when feasible (`R = mid`) and discard it when
   not (`L = mid + 1`); this lower-bound form needs no sentinel and cannot overshoot.
3. *Overflow.* `m*k` can reach `2*10^14` — test impossibility with `__int128`. `lo + hi` can reach
   `2*10^9` — take the midpoint as `lo + (hi - lo)/2`, never `(lo+hi)/2`. Keep days in `long long`.

**Edge cases.** `m*k > n` -> `-1`; `k = 1` -> the `m`-th smallest bloom day; single bed -> its day
(loop body never runs); all beds same day -> that day; a run broken by a late bloomer forces a
strictly later day (so the answer is not merely the `m`-th smallest day — adjacency matters); a run
already complete at `min b` returns `min b`.

**Complexity.** `O(n log(maxDay))` time, `O(n)` space (the bloom-day array).

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long m, k;
    if (scanf("%d %lld %lld", &n, &m, &k) != 3) return 0;
    vector<long long> b(n);
    for (int i = 0; i < n; i++) scanf("%lld", &b[i]);

    // We want the earliest day t on which at least m bouquets can be assembled.
    // A bed i is bloomed on day t iff b[i] <= t (it blooms on day b[i] and stays).
    // A bouquet needs k consecutive bloomed beds; a maximal bloomed run of length
    // L yields floor(L/k) bouquets. Need the total >= m.

    // Impossibility: even with every bed bloomed we have n beds, giving at most
    // floor(n/k) bouquets. If m*k > n, it can never be done.
    if ((__int128)m * (__int128)k > (__int128)n) {
        printf("-1\n");
        return 0;
    }

    // The earliest feasible day is necessarily one of the bloom days, so the
    // search range is [min b, max b], inclusive on both ends.
    long long lo = LLONG_MAX, hi = LLONG_MIN;
    for (int i = 0; i < n; i++) { lo = min(lo, b[i]); hi = max(hi, b[i]); }

    auto feasible = [&](long long t) -> bool {
        long long bouquets = 0, run = 0;
        for (int i = 0; i < n; i++) {
            if (b[i] <= t) {                 // bloomed by day t (inclusive)
                run++;
                if (run == k) { bouquets++; run = 0; }
                if (bouquets >= m) return true;
            } else {
                run = 0;
            }
        }
        return bouquets >= m;
    };

    // Binary search for the minimal feasible t in [lo, hi].
    // feasible(hi) is true because on day hi every bed is bloomed => one run of
    // length n => floor(n/k) >= m (from the m*k <= n check above).
    long long L = lo, R = hi;
    while (L < R) {
        long long mid = L + (R - L) / 2;     // floors toward L; no overflow
        if (feasible(mid)) R = mid;          // mid works: answer in [L, mid]
        else L = mid + 1;                    // mid fails: answer in [mid+1, R]
    }
    printf("%lld\n", L);
    return 0;
}
```
