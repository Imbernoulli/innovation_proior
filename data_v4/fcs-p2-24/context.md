# Minimum number of perfect squares summing to n

## Research question

Given a positive integer `n`, write it as a sum of perfect-square numbers
(`1, 4, 9, 16, 25, ...`) using **as few terms as possible**, and output that
minimum count. Squares may be reused (e.g. `12 = 4 + 4 + 4`), and there is no
limit on how many terms you use other than minimizing the count.

Every integer `n >= 1` has at least the trivial decomposition `n = 1 + 1 + ... + 1`
(`n` ones), so an answer always exists; the task is to find the smallest number
of squares. This is a classic "fewest coins" shaped optimization where the
"coins" are the perfect squares, and the catch is that the locally-greedy choice
of "take the largest square that fits" is **not** guaranteed to be optimal.

## Input / output contract

- Input (stdin): a single integer `n` with `0 <= n <= 10^6`.
- Output (stdout): a single line with the minimum number of perfect squares that
  sum to exactly `n`. For `n = 0` the answer is `0` (the empty sum).
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `n = 12` the answer is `3`, because `12 = 4 + 4 + 4` (three squares),
and no decomposition into two squares exists.

## Background

The decomposition constraint makes this a minimization over a structured set of
"denominations" (the perfect squares). Two routes are on the table before
committing:

- **Greedy by largest square.** Repeatedly subtract the largest perfect square
  not exceeding the remaining amount, counting terms, until the remainder is `0`.
  It is `O(sqrt(n))` per query and trivial to write; the open question is whether
  always grabbing the biggest fitting square is actually optimal. (Unlike coin
  systems such as US currency, the squares are **not** a "canonical" system, so
  this needs to be checked rather than assumed.)
- **Bottom-up dynamic programming.** Compute `dp[v]` = fewest squares summing to
  `v` for every `v` from `0` to `n`, via `dp[v] = 1 + min over squares s*s <= v
  of dp[v - s*s]`. This is `O(n * sqrt(n))`; the open question is the exact
  recurrence, the base case, and whether it fits the time limit at `n = 10^6`.

## Evaluation settings

Judged on hidden tests covering: small values (`n = 0, 1, 2, 3, ...`), perfect
squares themselves (answer `1`), values that are the known hardest case of "four
squares needed" (numbers of the form `4^a (8b + 7)`, e.g. `7, 15, 23, 28, 31`),
values where greedy overshoots the optimum (e.g. `12`), and the maximum
`n = 10^6` (to confirm the chosen method fits in time and memory).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;

    // TODO: compute the minimum number of perfect squares summing to n.
    int answer = 0;

    cout << answer << "\n";
    return 0;
}
```
