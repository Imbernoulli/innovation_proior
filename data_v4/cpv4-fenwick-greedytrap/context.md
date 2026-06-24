# Maximum-sum strictly increasing subsequence

## Research question

A trader records `n` daily prices `a[0..n-1]` of a single asset (prices may be negative — think of
mark-to-market values that can dip below the cost basis). On a chosen set of days she opens a position,
but her strategy only allows entering on days whose prices form a **strictly increasing subsequence**
in time order: if she enters on days `i1 < i2 < ... < ik` then `a[i1] < a[i2] < ... < a[ik]`. Her score
is the **sum** of the chosen prices. She must enter on at least one day. Output the maximum achievable
score.

This is the *maximum-sum strictly increasing subsequence* problem. It looks like the longest-increasing-
subsequence (LIS) family, but the objective is the **sum**, not the length, and that difference is the
whole trap: the longest increasing chain and the heaviest increasing chain are frequently different
chains. Getting it right at scale forces an order-statistic data structure — a Fenwick tree indexed by
value-rank that holds a running **prefix maximum** of partial scores.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2*10^5`); then `n` integers `a[i]`
  (`-10^9 <= a[i] <= 10^9`), whitespace-separated.
- Output (stdout): a single line with the maximum sum of a non-empty strictly increasing subsequence.
  If `n = 0` there is no day to enter, and by convention the score is `0`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `a = [3, 1, 4, 1, 5, 9, 2, 6]` the answer is `21`, achieved by the increasing subsequence
`3 < 4 < 5 < 9` (sum `3 + 4 + 5 + 9 = 21`).

## Background

The constraint "strictly increasing in value, increasing in index" makes this a constrained selection
problem, and two families of approach are on the table before committing.

- **Greedy.** Two tempting greedy heuristics suggest themselves: take the *longest* increasing
  subsequence (more terms must mean a bigger sum), or just take the *single largest* value. Both are
  `O(n log n)` and trivial; the open question is whether maximizing the count — or grabbing one big
  element — actually maximizes the sum under the ordering constraint.
- **Value-indexed DP accelerated by a Fenwick tree.** Define `f[i]` = best score of an increasing
  subsequence that *ends* at index `i`. Then `f[i] = a[i] + max(0, max{ f[j] : j < i, a[j] < a[i] })`.
  The naive scan over `j` is `O(n^2)`. Because the inner query is "maximum `f` over all earlier elements
  with strictly smaller value," coordinate-compress the values and keep a Fenwick tree that supports
  prefix-maximum query and point update, giving `O(n log n)`. The open questions are the exact recurrence,
  the strict-vs-non-strict boundary in the prefix query, and the duplicate-value handling.

## Evaluation settings

Judged on hidden tests covering: strictly increasing arrays (answer is the full sum), strictly
decreasing arrays (answer is the single largest element), arrays with many duplicate values (strictness
must exclude equal predecessors), all-negative arrays (answer is the least-negative single element), the
empty array (`n = 0`, answer `0`), single element (`n = 1`), and large `n = 2*10^5` with values near
`10^9` (so the running sum can exceed a 32-bit integer).

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
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // TODO: compute the maximum sum of a non-empty strictly increasing subsequence.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
