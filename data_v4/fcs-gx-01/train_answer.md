**Problem.** Schedule `n` jobs on a single machine (one at a time, no idle, no preemption). Job `i`
takes `t[i]` time and has weight `w[i]`. For an order, the completion time `C[i]` is the prefix sum
of processing times up to and including job `i`, and the schedule cost is `sum_i w[i] * C[i]`. Read
`n` and the pairs `(t[i], w[i])` from stdin; print the minimum achievable total weighted completion
time. This is `1 || sum w_j C_j`.

**Why the obvious keys are wrong.** Both single-key sorts ignore half the data, because the cost
`w[i]*C[i]` couples length and weight. *Shortest-job-first* (sort by `t` ascending) is optimal only
when all weights are equal; *heaviest-first* (sort by `w` descending) is optimal only when all
lengths are equal. On `A=(t=1,w=1), B=(t=3,w=5), C=(t=2,w=1)`, shortest-job-first gives order
`A,C,B` with cost `1*1 + 1*3 + 5*6 = 34`, but pulling the heavy `B` forward, order `B,A,C`, costs
`5*3 + 1*4 + 1*6 = 25`. Each unit of delay on `B` costs 5x, so finishing it early is worth delaying
the light jobs. Both single keys are discarded.

**Key idea — Smith's rule from an exchange argument.** Take two adjacent jobs `i` (first) then `j`,
with time `P` already accumulated before them. Comparing the two local orders, the `P`-terms and the
`w*t` self-terms cancel, and the cost difference is exactly

```
(i before j) - (j before i) = w[j]*t[i] - w[i]*t[j].
```

So `i` belongs before `j` iff `t[i]*w[j] <= t[j]*w[i]`, i.e. ascending ratio `t/w`. Because this is
a consistent total order, the sorted schedule admits no improving adjacent swap and is therefore
globally optimal. Sort by this coupled comparator in `O(n log n)`, then sweep once accumulating
`clock += t[i]; cost += w[i]*clock`.

**Pitfalls to get right.**
1. *Strict weak ordering on ties.* Many jobs can share a ratio (cross-product difference 0). The
   comparator must be strict (`<`) with a deterministic tie-break (by index); a `<=` form makes both
   `cmp(i,j)` and `cmp(j,i)` true on a tie, which is undefined behavior in `std::sort` and can crash
   or scramble output. Ties are cost-invariant, so the tie-break does not change the answer.
2. *No floats.* Compare cross-multiplied integers `t[i]*w[j]` vs `t[j]*w[i]` (each `<= 10^8`), never
   the `double` ratio `t/w`, which can collide on near-equal ratios and violate the ordering
   contract.
3. *Overflow.* The running `clock` reaches `~2*10^9` and the total cost reaches `~2*10^18`; use
   `long long` for both. A 32-bit accumulator is a silent wrong answer on the large tests.

**Edge cases (all handled).** `n = 0` -> cost `0` (loops never run). `n = 1` -> the single forced
completion time. All weights equal -> comparator degenerates to shortest-job-first. All times equal
-> comparator degenerates to heaviest-first. Many equal ratios -> tie-break keeps the sort defined,
cost unchanged.

**Complexity.** `O(n log n)` time for the sort, `O(n)` extra space for the index array; the sweep is
`O(n)`.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> cost 0

    vector<long long> t(n), w(n);
    for (int i = 0; i < n; i++) cin >> t[i] >> w[i];

    // Single machine, minimize sum of w[i] * C[i], where C[i] is the completion
    // time (prefix sum of processing times) of job i in the chosen order.
    //
    // Exchange argument: for two adjacent jobs i (first) then j (second), the part
    // of the cost that depends on their relative order is, letting P be the time
    // accumulated before the pair,
    //     i before j:  w[i]*(P+t[i]) + w[j]*(P+t[i]+t[j])
    //     j before i:  w[j]*(P+t[j]) + w[i]*(P+t[j]+t[i])
    // The P-terms and the w*t self-terms cancel; the difference reduces to
    //     (i before j) - (j before i) = w[j]*t[i] - w[i]*t[j].
    // So putting i first is no worse exactly when w[j]*t[i] <= w[i]*t[j], i.e.
    //     t[i]*w[j] <= t[j]*w[i]   (Smith's ratio rule: ascending t/w).
    // Sorting by this total order is therefore optimal. Use cross-multiplication
    // (not the float ratio t/w) to stay exact; products fit in long long since
    // t,w <= 1e4 -> product <= 1e8, and the total cost <= ~2e18 < 9.2e18.
    vector<int> idx(n);
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&](int i, int j) {
        // i before j iff t[i]*w[j] < t[j]*w[i]; ties keep a stable, total order.
        long long lhs = t[i] * w[j];
        long long rhs = t[j] * w[i];
        if (lhs != rhs) return lhs < rhs;
        return i < j;                      // deterministic tie-break (cost is tie-invariant)
    });

    long long cost = 0;     // total weighted completion time, <= ~2e18, fits in signed 64-bit
    long long clock = 0;    // running completion time (prefix sum of processing times)
    for (int k = 0; k < n; k++) {
        int i = idx[k];
        clock += t[i];                     // completion time of job i
        cost += w[i] * clock;
    }

    cout << cost << "\n";
    return 0;
}
```
