**Problem.** A building has `m` floors and there is a critical floor `c` such that an egg breaks
when dropped from any floor `>= c` and survives below it. With `k` eggs (broken eggs are gone,
survivors are reusable), find a dropping strategy guaranteed to identify `c` and report the minimum
number of trials its worst case needs. Read `k` and `m` from stdin; print that minimum.
Constraints: `1 <= k <= 100`, `1 <= m <= 10000`.

**Why the obvious binary search is wrong.** "Drop from the middle and halve the range" gives about
`ceil(log2(m+1))` trials — optimal only with *unlimited* eggs. With limited eggs it is not just
sub-optimal, it can be *infeasible*: for `k = 1, m = 100`, binary search drops from floor 50, and if
the egg breaks there are zero eggs left and `c` is still unknown — the strategy cannot finish. The
only guaranteed one-egg strategy is the linear scan (floors `1, 2, 3, …`), worst case `m = 100`, not
`7`. Limited eggs cap how aggressively you may break, so binary search is discarded.

**Key idea — a dual minimax DP over (eggs, trials).** Instead of "fewest trials for `f` floors,"
ask "most floors coverable with `e` eggs and `t` trials," `cover(e, t)`. One optimal drop splits the
work into the below-part (if it breaks: `e-1` eggs, `t-1` trials), the above-part (if it survives:
`e` eggs, `t-1` trials), and the drop floor itself:

- `cover(e, t) = cover(e-1, t-1) + cover(e, t-1) + 1`, with `cover(e, 0) = 0`, `cover(0, t) = 0`.

`cover` is increasing in `t`, so the answer is the smallest `t` with `cover(k, t) >= m`. This needs
no inner minimisation (the `+1` recurrence already encodes the best balanced drop), giving a clean,
provable `O(k * t_answer)` loop. The primal `(eggs, floors)` DP
`T(e,f) = 1 + min_x max(T(e-1,x-1), T(e,f-x))` is equally correct but `O(k * m^2)` — too slow here —
so it serves only as an independent oracle.

**Two things to get right.**
1. *Update order.* Advancing one trial in place, `cover[e] = cover[e-1] + cover[e] + 1` must read
   *previous-trial* values for both terms. Sweeping `e` upward overwrites `cover[e-1]` first and
   feeds a same-trial value into `cover[e]` (a trace of `kk=2` yields `cover(2,1)=2` instead of `1`).
   Sweep `e` from high to low so `cover[e-1]` is still last trial's value.
2. *Egg cap and overflow.* With `t` trials any egg count covers at most `2^t - 1` floors, and
   `2^14 - 1 = 16383 >= 10000`, so the answer never exceeds 14 once `k >= 14`; an egg beyond the
   14th can never be dropped, so capping the working egg count at `min(k, 14)` is exact. Clamping
   every `cover` value at `m` keeps everything tiny and removes any overflow concern.

**Edge cases.** `m = 1 -> 1` for every `k`; `k = 1 -> m` (linear scan); `k >= 14` collapses to the
unlimited-egg `ceil(log2(m+1))`; the loop terminates because `cover[k]` rises by at least 1 each
trial.

**Validation.** The dual `cover` solution was differential-tested against the independent primal
floors DP over 700+ random and edge cases (zero mismatches), against a closed-form
`cover(k,t) = sum_{i=1..k} C(t,i)` at the extreme corners (`m = 10000`, `k in {1,2,3,13,14,50,100}`),
and against every classic textbook value (`k=2,m=6 -> 3`; `k=3,m=14 -> 4`; `k=2,m=100 -> 14`),
running in a couple of milliseconds per case.

**Complexity.** `O(k * t_answer)` time, bounded by `O(min(k,14) * m) <= ~1.4*10^5`; `O(min(k,14))`
extra space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int k;       // number of eggs, 1 <= k <= 100
    long long m; // number of floors, 1 <= m <= 10000
    if (!(cin >> k >> m)) return 0;

    // cover[e] = maximum number of floors distinguishable with e eggs and the
    // current number of trials t. Recurrence over one more trial:
    //   cover_t(e) = cover_{t-1}(e-1) + cover_{t-1}(e) + 1
    // (drop from a floor: if the egg breaks we have e-1 eggs and t-1 trials for
    //  the floors below; if it survives we have e eggs and t-1 trials for the
    //  floors above; plus the floor we dropped from). cover_0(e) = 0 for all e,
    //  and cover_t(0) = 0 for all t. We want the smallest t with cover_t(k) >= m.
    //
    // We cap the working egg count at min(k, 14): with t trials the most floors
    // any number of eggs can cover is 2^t - 1, and 2^14 - 1 = 16383 >= 10000 = m,
    // so the answer never exceeds 14 once eggs >= 14. With at most 14 trials an
    // egg beyond the 14th can never be dropped, so eggs past 14 are irrelevant
    // and capping the working egg count at 14 leaves the answer unchanged.
    int kk = (int)min<long long>(k, 14);
    vector<long long> cover(kk + 1, 0);
    long long t = 0;
    while (cover[kk] < m) {
        ++t;
        // update from high egg count to low so cover[e-1] still holds the
        // previous-trial value when we read it.
        for (int e = kk; e >= 1; --e) {
            long long below = cover[e - 1];           // breaks: e-1 eggs, t-1 trials
            long long above = cover[e];               // survives: e eggs, t-1 trials
            long long c = below + above + 1;
            if (c > m) c = m;                          // clamp to avoid overflow
            cover[e] = c;
        }
    }

    cout << t << "\n";
    return 0;
}
```
