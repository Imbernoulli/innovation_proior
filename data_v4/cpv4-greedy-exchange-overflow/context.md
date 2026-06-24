# Minimizing total weighted completion time on one machine (Smith's rule)

## Research question

A single repair bench processes `n` jobs **one at a time**, with no idle gaps and no preemption: once
you fix an ordering, the jobs run back to back in that order. Job `i` needs `p[i]` units of bench time
to finish and carries a weight `w[i]` (think of it as a per-unit-of-delay penalty rate for that
customer). If a job *completes* at time `C` (the moment its processing ends), it contributes
`w[i] * C` to the total bill.

You choose the order of all `n` jobs. Output the **minimum** achievable value of
`sum over i of w[i] * C[i]`, where `C[i]` is the completion time of job `i` under your chosen order.

This is the classic `1 || sum w_j C_j` scheduling objective. Every job is eventually run (you are not
selecting a subset — you are ordering the whole set), so the only freedom is the permutation. The
question is which permutation minimizes the weighted sum of completion times, and how to compute the
resulting cost without tripping over the very large intermediate quantities.

## Input / output contract

- Input (stdin):
  - the first token is `n` (`0 <= n <= 2*10^4`);
  - then `n` integers `p[0..n-1]` (`1 <= p[i] <= 10^5`), the processing times;
  - then `n` integers `w[0..n-1]` (`1 <= w[i] <= 10^5`), the weights.
  - All tokens are whitespace-separated; line breaks are not significant.
- Output (stdout): a single line with the minimum total weighted completion time.
- Time limit: 1 second. Memory: 256 MB.

Example: for `p = [3, 1, 2]`, `w = [1, 2, 3]` the answer is `17`.

## Background

The objective is *separable over an ordering*: fix any permutation and the cost is the sum, over each
job in turn, of (its weight) times (the running clock once it finishes). Two families of approach are
on the table before committing to one:

- **Greedy exchange (sequencing rule).** Sort the jobs by a single per-job key and run them in that
  order. The candidate key is the ratio `p[i] / w[i]` (shortest *weighted* processing time first).
  The exchange argument: if two jobs that are out of ratio order sit next to each other, swapping them
  cannot increase the total, so an optimal order is sorted by the ratio. Cost is `O(n log n)`. The
  open questions are (a) proving the local swap argument really gives a global optimum and (b)
  comparing ratios `p[i]/w[i]` without floating-point error — which forces the cross-multiplied form
  `p[i] * w[j]` vs `p[j] * w[i]`, whose products are large.
- **Permutation search / DP over subsets.** Try orderings directly. Exhaustive permutation is `O(n!)`
  and a bitmask DP is `O(2^n n)`; both are fine as an independent correctness oracle on tiny `n` but
  hopelessly slow at `n = 2*10^4`. Useful only for cross-checking the greedy.

## Evaluation settings

Judged on hidden tests covering: `n = 0` (empty, answer `0`); `n = 1`; many jobs with equal ratios
(ties, where any consistent order is optimal); jobs whose ratios are close so the cross-multiplied
comparison must be exact; and large instances with `n = 2*10^4` and `p[i], w[i]` near `10^5`, so that
(a) the comparison product `p[i] * w[j]` reaches `~10^10` and (b) the accumulated answer reaches
`~2*10^18` — both far outside 32-bit range.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<long long> p(n), w(n);
    for (int i = 0; i < n; i++) cin >> p[i];
    for (int i = 0; i < n; i++) cin >> w[i];

    // TODO: order the jobs by Smith's rule and accumulate the minimum
    //       total weighted completion time sum_i w[i] * C[i].
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
