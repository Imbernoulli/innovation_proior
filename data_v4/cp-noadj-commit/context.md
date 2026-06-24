# Maximum-weight independent set on a path (no two adjacent)

## Research question

You are given a sequence of `n` integers `a[0..n-1]` (values may be negative). Choose a subset of the
positions so that **no two chosen positions are adjacent** (you may choose none), and **maximize the
sum** of the chosen values. Output that maximum sum. Because the empty subset is allowed, the answer
is always at least `0`.

This is the path case of maximum-weight independent set. It is the kind of subproblem that shows up
inside scheduling, interval, and DP-on-tree problems, so getting the one-dimensional version exactly
right — including the negative-value and empty-subset corners — matters.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2*10^5`); then `n` integers `a[i]`
  (`-10^9 <= a[i] <= 10^9`), whitespace-separated.
- Output (stdout): a single line with the maximum achievable sum.
- Time limit: 1 second. Memory: 256 MB.

Example: for `a = [5, 1, 1, 5, 1, 5]` the answer is `15` (take indices 0, 3, 5).

## Background

The constraint "no two chosen adjacent" makes this a constrained selection problem. Two families of
approach are on the table before committing to one:

- **Greedy by value.** Repeatedly take the largest remaining positive element and forbid its two
  neighbours. Greedy is `O(n log n)` and trivial to write; the open question is whether always
  grabbing the largest element is actually optimal under the adjacency constraint.
- **Linear dynamic programming.** Scan left to right maintaining, for each prefix, the best sum that
  ends with the last element *taken* versus *not taken*. This is `O(n)`; the open question is the
  exact recurrence and how the transitions reference the previous state.

## Evaluation settings

Judged on hidden tests covering: all-positive arrays, arrays with negatives and zeros, the empty
array (`n = 0`), single element (`n = 1`), all-negative arrays (answer should be `0`), and large
`n = 2*10^5` with values near `10^9` (so the running sum can exceed a 32-bit integer).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // TODO: compute the maximum sum of a subset with no two adjacent positions (empty allowed).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
