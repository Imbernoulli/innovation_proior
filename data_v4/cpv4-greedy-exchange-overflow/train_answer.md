**Problem.** `n` jobs run one at a time on a single bench, no gaps and no preemption; you choose the order of *all* jobs. Job `i` needs `p[i]` time and has weight `w[i]`, and if it completes at time `C[i]` it costs `w[i] * C[i]`. Read `n`, then the `p` array, then the `w` array from stdin; print the minimum possible `sum_i w[i] * C[i]`.

**Key idea — Smith's rule (greedy exchange).** Order the jobs by the ratio `p[i] / w[i]` ascending and run them in that order. *Why:* take any two adjacent jobs `A` (just before) and `B`. Everything outside this pair keeps its completion time when you swap them, because the block `{A,B}` occupies the same time window either way. Only the two terms for `A` and `B` change, and (A-first cost) − (B-first cost) `= w_B p_A − w_A p_B`. So A-first is no worse than B-first exactly when `p_A w_B <= p_B w_A`, i.e. `p_A/w_A <= p_B/w_B`. Any non-sorted order has an adjacent inversion you can fix without increasing cost, so the ratio-sorted order is globally optimal. Then sweep once, keeping a running clock; when job `i` finishes, the clock equals `C[i]` and you add `w[i] * clock`.

**Comparing ratios without floating point.** Compare `p[i]/w[i] < p[j]/w[j]` as the cross-multiplied `p[i] * w[j] < p[j] * w[i]` — no division, no precision loss. Equal products mean equal ratios (a legal "equivalent" pair for `std::sort`); add an `i < j` tie-break for deterministic output. Ties are free: the swap difference is exactly zero, so any order of equal-ratio jobs gives the same cost.

**Pitfalls.**
1. *Integer overflow (the headline trap).* With `p[i], w[i] <= 10^5`, the comparator product `p[i] * w[j]` reaches `10^10`; the running clock reaches `n * max(p) = 2*10^9` (already past 32-bit); the final answer reaches `~2*10^18`. A 32-bit `int` anywhere in a product or sum silently overflows. The comparator failure is the nasty one: `60000 * 90000 = 5.4*10^9` wraps to `1,105,032,704` in `int`, flipping the comparison and *corrupting the sort order*, so the answer is grossly wrong (an int build prints `1.16*10^16` where the truth is `9.38*10^15`). Declare `p`, `w`, the clock, and the accumulator as `long long`.
2. *Dividing to compare ratios* would introduce floating error on near-equal ratios; use the integer cross-multiplication instead.
3. *Unstable/invalid comparator* — `p[i]*w[j] < p[j]*w[i]` is a strict weak ordering (equal products compare equivalent), so it is valid; the explicit tie-break only fixes the output, it does not affect cost.

**Edge cases.** `n = 0` → no jobs, answer `0` (loops run zero times). `n = 1` → only one order, answer `w[0]*p[0]`. All equal ratios → order-independent; the max-magnitude instance (`n = 2*10^4`, all `p = w = 10^5`) gives exactly `2000100000000000000`, which fits in `long long`.

**Complexity.** `O(n log n)` time for the sort, `O(n)` extra space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // no jobs -> total cost 0
    vector<long long> p(n), w(n);
    for (int i = 0; i < n; i++) cin >> p[i];
    for (int i = 0; i < n; i++) cin >> w[i];

    vector<int> idx(n);
    iota(idx.begin(), idx.end(), 0);
    // Smith's rule: order by p/w ascending. Compare p_i/w_i < p_j/w_j as a
    // cross-multiplication p_i * w_j < p_j * w_i with NO division. Both products
    // can reach 1e5 * 1e5 = 1e10, which overflows 32-bit; p and w are long long.
    sort(idx.begin(), idx.end(), [&](int i, int j) {
        long long lhs = p[i] * w[j];
        long long rhs = p[j] * w[i];
        if (lhs != rhs) return lhs < rhs;
        return i < j;                       // deterministic tie-break
    });

    long long clock = 0;                    // running completion time (up to 2e9)
    long long answer = 0;                   // sum of w_i * C_i (up to ~2e18)
    for (int k = 0; k < n; k++) {
        int i = idx[k];
        clock += p[i];                      // this job finishes at 'clock'
        answer += w[i] * clock;             // weighted completion time
    }

    cout << answer << "\n";
    return 0;
}
```
