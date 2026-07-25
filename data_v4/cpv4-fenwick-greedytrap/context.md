# Maximum-sum strictly increasing subsequence

## Research question

A trader records `n` daily prices `a[0..n-1]` of a single asset (prices may be negative — think of
mark-to-market values that can dip below the cost basis). On a chosen set of days she opens a position,
but her strategy only allows entering on days whose prices form a **strictly increasing subsequence**
in time order: if she enters on days `i1 < i2 < ... < ik` then `a[i1] < a[i2] < ... < a[ik]`. Her score
is the **sum** of the chosen prices. She must enter on at least one day. Output the maximum achievable
score.

This is the *maximum-sum strictly increasing subsequence* problem.

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
problem.

- **Greedy.** Two tempting greedy heuristics suggest themselves: take the *longest* increasing
  subsequence (more terms must mean a bigger sum), or just take the *single largest* value. Both are
  `O(n log n)` and trivial; the open question is whether maximizing the count — or grabbing one big
  element — actually maximizes the sum under the ordering constraint.

## Evaluation settings

Judged on hidden tests covering: strictly increasing arrays, strictly decreasing arrays, arrays with
many duplicate values, all-negative arrays, the empty array (`n = 0`), single element (`n = 1`), and
large `n = 2*10^5` with values near `10^9`.

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
