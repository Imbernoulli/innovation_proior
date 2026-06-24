# Fewest perfect squares summing to N

## Research question

You are given a single positive integer `N`. A *square term* is any number of the form `k*k`
with `k >= 1`, i.e. one of `1, 4, 9, 16, 25, ...`. You may use the same square value more than
once. Choose a multiset of square terms whose sum is **exactly** `N`, using **as few terms as
possible**, and output that minimum count.

For example `N = 12` can be written as `4 + 4 + 4` (three terms) but not as one or two squares,
so the answer is `3`. This is the classic "represent `N` as a sum of squares" question, and it sits
at the boundary between an innocent-looking greedy, a textbook dynamic program, and a sharp
number-theoretic shortcut — which one you reach for decides whether the solution survives the
constraints.

## Input / output contract

- Input (stdin): a single integer `N` (`1 <= N <= 10^9`).
- Output (stdout): a single line with the minimum number of square terms that sum to exactly `N`.
- Time limit: 1 second. Memory: 256 MB.

Examples:
- `N = 12` -> `3`  (`4 + 4 + 4`)
- `N = 7`  -> `4`  (`4 + 1 + 1 + 1`)
- `N = 13` -> `2`  (`9 + 4`)
- `N = 25` -> `1`  (`25`)

## Background

Three families of approach present themselves, and the constraints decide between them:

- **Greedy by largest square.** Repeatedly subtract the largest square `<= ` the remaining amount
  and count the steps. This is `O(sqrt(N))` and a four-line loop. The open question is whether
  grabbing the biggest square at each step is actually optimal — greedy is famously correct for some
  coin systems and wrong for others, and "squares" is not an obvious case either way.
- **Dynamic programming over amounts.** Define `dp[v]` = fewest squares summing to `v`, with
  `dp[v] = 1 + min over squares s<=v of dp[v - s]`. This is exact and easy to trust, but it is
  `O(N * sqrt(N))` time and `O(N)` memory. The open question is whether it fits when `N` can be as
  large as `10^9`.
- **Number theory.** Lagrange's four-square theorem guarantees the answer is always `1, 2, 3,` or
  `4`. If that is true, the entire problem collapses to *deciding which of the four* — a constant
  number of cheap tests — and `N`'s size stops mattering. The open question is the exact decision
  rule for `3` versus `4`.

## Evaluation settings

Judged on hidden tests covering: perfect squares (answer `1`), sums of two squares (answer `2`),
numbers of the special form `4^k * (8m + 7)` (answer `4`), generic numbers (answer `3`), the
smallest inputs (`N = 1, 2, 3`), and large `N` near `10^9` where any `O(N)` method would time out
or exhaust memory and where 32-bit intermediate squares would overflow.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    long long n;
    if (!(cin >> n)) return 0;

    // TODO: output the minimum number of perfect squares (k*k, k>=1) summing to exactly n.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
