# Minimum-cost assignment of n workers to n tasks

## Research question

You are given an `n x n` cost matrix. `cost[i][j]` is the cost of giving worker `i` task `j`.
Assign **every** worker to **exactly one** task and **every** task to **exactly one** worker
(a perfect matching / permutation), so that the **total cost is minimized**. Output that minimum
total cost.

This is the classic *assignment problem*. It appears as a subroutine in scheduling, resource
allocation, and matching pipelines, so getting the exact optimum — not a plausible-looking
heuristic value — is what matters here.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 18`); then `n*n` integers giving the matrix in
  row-major order, so the `i`-th row is `cost[i][0], cost[i][1], ..., cost[i][n-1]`. Costs satisfy
  `-10^9 <= cost[i][j] <= 10^9` and are whitespace-separated.
- Output (stdout): a single line with the minimum achievable total cost.
- For `n = 0` the answer is `0` (no workers, no tasks, empty assignment).
- Time limit: 2 seconds. Memory: 256 MB.

Example: for

```
3
0 6 3
6 0 8
3 7 7
```

the answer is `6` (worker 0 -> task 2 = 3, worker 1 -> task 1 = 0, worker 2 -> task 0 = 3).

## Background

A valid assignment is a permutation `p` of `{0,...,n-1}` with total cost `sum_i cost[i][p[i]]`; we
want the minimum over all `n!` permutations. Two families of approach are on the table before
committing to one:

- **Cheapest-available greedy.** Repeatedly take the smallest remaining cell `cost[i][j]` whose
  worker `i` and task `j` are both still free, then forbid that worker and that task; or, the
  row-by-row variant where worker `i` grabs its cheapest still-free task. Both are `O(n^2 log n)` and
  a few lines. The open question is whether locally grabbing the cheapest cell is actually optimal
  under the perfect-matching constraint.
- **Bitmask dynamic programming over the set of assigned tasks.** Place workers in index order; the
  state is the *set* of tasks already used. This is `O(n^2 * 2^n)`; the open question is the exact
  recurrence and whether `2^n` is affordable at `n <= 18`.

## Evaluation settings

Judged on hidden tests covering: matrices where the cheapest-available greedy is strictly
suboptimal, all-equal matrices, matrices with negative and zero costs, the empty matrix (`n = 0`),
a single worker (`n = 1`, including a negative cost), tie-heavy matrices, and the largest case
`n = 18` with costs near `10^9` (so the total can exceed a 32-bit integer).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<vector<long long>> cost(n, vector<long long>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cin >> cost[i][j];

    // TODO: compute the minimum total cost of a perfect worker->task assignment.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
