**Problem.** A single repair bench runs `n` jobs back-to-back with no idle time and no preemption. Job `i` needs `t[i]` minutes and is charged `w[i]` per minute until *it* finishes; in a given order, job `i` finishes at `C[i]` = the running sum of `t` up to and including it, contributing `w[i] * C[i]`. Choose the order to minimize `sum_i w[i] * C[i]` (the single-machine total weighted completion time problem, `1 || sum w_j C_j`). Read `n` then the `n` pairs `(t[i], w[i])` from stdin; print the minimum.

**Why the obvious greedies are wrong.** Two single-attribute keys look right and both fail. *Shortest first* (sort by `t` ascending) ignores the weights; *most impatient first* (sort by `w` descending) ignores the durations. On the three jobs `A=(t=3,w=4)`, `B=(t=1,w=1)`, `C=(t=2,w=3)`: shortest-first runs `B,C,A` for cost `34`, impatient-first runs `A,C,B` for cost `33`, but the optimum is `32` via `C,A,B`. Each greedy throws away exactly one attribute the weighted objective depends on, so both are discarded. (Each is merely the special case of the correct rule when the *other* attribute is constant — which is why each looked plausible.)

**Key idea — Smith's rule via an exchange argument.** Take any order and two *adjacent* jobs `i` then `j`, with total duration `P` of everything before them. Running `i,j` costs `w_i(P+t_i) + w_j(P+t_i+t_j)`; running `j,i` costs `w_j(P+t_j) + w_i(P+t_j+t_i)`. Everything outside the pair is unchanged (jobs before see the same `P`, jobs after see the same `P+t_i+t_j`). The difference is

`(run i,j) - (run j,i) = w_j t_i - w_i t_j`.

So `i` before `j` is no worse exactly when `w_j t_i - w_i t_j <= 0`, i.e. `t_i / w_i <= t_j / w_j`. Sorting by `t/w` ascending therefore removes every cost-reducing inversion, so it is optimal. Then scan the sorted order, accumulate the running completion time `cur`, and add `w[i] * cur` each step. `O(n log n)`.

**Pitfalls.**
1. *Wrong key.* Sort by the *ratio* `t/w`, not the *product* `t*w`. Compare with the cross-multiplied form `t[i]*w[j] < t[j]*w[i]` (crossed indices) to stay in exact integers — no floating-point, no division. Sorting by `t*w` instead silently reproduces shortest-first on `A,B,C` and returns the suboptimal `34`.
2. *Strict weak ordering.* Equal-ratio jobs satisfy `t[i]*w[j] == t[j]*w[i]`, so the comparator returns `false` both ways — they are incomparable, which is what `std::sort` needs, and the exchange difference is `0` so their relative order is free.
3. *Overflow.* Completion times reach `~10^9` and the weighted sum reaches `~5*10^17`; use `long long` for the accumulators and for `t,w` (so comparator products `t*w <= 10^8` and `w*cur <= 10^13` stay 64-bit). An `int` is a silent wrong-answer on the large tests.

**Edge cases.** `n = 0` -> empty sort and empty loop -> `0`. `n = 1` -> the lone job finishes at `t[0]`, cost `w[0]*t[0]`. All equal durations -> rule degenerates to impatient-first (optimal there); all equal weights -> degenerates to shortest-first (optimal there); all equal ratios -> every order costs the same.

**Complexity.** `O(n log n)` time for the sort, `O(n)` extra space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> t(n), w(n);
    for (int i = 0; i < n; i++) cin >> t[i] >> w[i];

    // Smith's rule: process jobs in non-decreasing order of t/w.
    // Compare i before j  <=>  t[i]*w[j] < t[j]*w[i]   (cross-multiply, all positive).
    vector<int> ord(n);
    iota(ord.begin(), ord.end(), 0);
    sort(ord.begin(), ord.end(), [&](int i, int j) {
        // primary key: t/w ascending  =>  t[i]*w[j] < t[j]*w[i]
        // products fit in long long: t,w <= 1e4 each => product <= 1e8
        long long lhs = t[i] * w[j];
        long long rhs = t[j] * w[i];
        if (lhs != rhs) return lhs < rhs;
        return i < j; // deterministic tie-break (does not affect cost)
    });

    long long cur = 0;      // running completion time
    long long total = 0;    // sum of w[i] * C[i]
    for (int idx : ord) {
        cur += t[idx];
        total += w[idx] * cur;
    }

    cout << total << "\n";
    return 0;
}
```
