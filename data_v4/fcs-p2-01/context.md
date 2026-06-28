# Minimum number of coins to reach a target sum

## Research question

You are given `n` coin denominations and a target value `S`. You have an **unlimited** supply of each
denomination. Choose a multiset of coins whose values sum to **exactly** `S`, using as **few coins as
possible**, and output that minimum count. If `S` cannot be formed from the given denominations,
output `-1`.

This is the classic "coin change — minimum coins" problem with *arbitrary* denominations (not a
canonical currency system). It appears as a subroutine in making-change, knapsack-style budgeting,
and any "fewest unit operations to reach a value" setting, so the denominations are deliberately
allowed to be an arbitrary set rather than a textbook `{1, 5, 10, 25}`.

## Input / output contract

- Input (stdin):
  - The first line contains two integers `n` and `S`
    (`1 <= n <= 100`, `0 <= S <= 10^6`).
  - The second line contains `n` integers `c[0..n-1]`, the denominations
    (`1 <= c[i] <= 10^6`), whitespace-separated. Denominations may repeat and may exceed `S`.
- Output (stdout): a single line with the minimum number of coins summing to exactly `S`,
  or `-1` if no such multiset exists.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for denominations `{1, 3, 4}` and `S = 6`, the answer is `2` (use `3 + 3`).

## Background

Two families of approach are on the table before committing to one.

- **Greedy by largest coin.** Repeatedly subtract the largest denomination that is `<= remaining`
  until the remainder is `0` (or report failure). It is essentially `O(S / c_min)` (or
  `O(S log n)` with a sorted scan) and a few lines of code. The open question is whether always
  grabbing the largest fitting coin actually yields the minimum count for an *arbitrary* set of
  denominations.
- **Dynamic programming over sums.** For every value `s` from `1` to `S`, compute the minimum coins
  to make `s` by relaxing each denomination `c` against `s - c`. This is `O(S * n)`; the open
  questions are the exact recurrence, the unreachable/`-1` handling, and whether `O(S * n)` is fast
  enough at the stated limits.

## Evaluation settings

Judged on hidden tests covering: the canonical greedy-killer denomination sets (e.g. `{1, 3, 4}`),
sets without a `1` coin (so many targets are unreachable), `S = 0` (answer `0`), denominations larger
than `S`, duplicate denominations, single-denomination sets, fully unreachable targets (answer `-1`),
and large instances with `S = 10^6` and `n = 100` to confirm the chosen method runs in time.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long S;
    if (!(cin >> n >> S)) return 0;
    vector<long long> c(n);
    for (auto &x : c) cin >> x;

    // TODO: compute the minimum number of coins (unlimited supply of each
    // denomination) summing to exactly S, or -1 if S is unreachable.
    long long answer = -1;

    cout << answer << "\n";
    return 0;
}
```
