# 0/1 knapsack with few items but astronomically large capacity

## Research question

You are given `n` items, each with a positive integer weight `w[i]` and a positive integer value
`v[i]`. You also have a knapsack of integer capacity `C`. Choose a subset of the items whose total
weight is at most `C`, and **maximize the total value** of the chosen subset (the empty subset is
allowed, giving value `0`). Output that maximum value.

This is the classic 0/1 knapsack, but the regime is unusual: the **number of items is tiny**
(`n <= 40`) while the **capacity is enormous** (`C` up to `10^18`) and individual weights/values are
large (`up to 10^9`). That combination rules out the two textbook reflexes — the capacity is far too
large for a weight-indexed table, and the few-items count is exactly what tempts a fast greedy or a
hand-rolled branch-and-bound. Getting an *exact, provable* answer in this regime is the point.

## Input / output contract

- Input (stdin):
  - the first line has two integers `n` and `C` (`0 <= n <= 40`, `0 <= C <= 10^18`);
  - then `n` lines follow, the `i`-th containing two integers `w[i]` and `v[i]`
    (`1 <= w[i] <= 10^9`, `1 <= v[i] <= 10^9`).
- Output (stdout): a single line with the maximum total value achievable with total weight `<= C`.
- Time limit: 2 seconds. Memory: 256 MB.

Example:

```
3 10
6 13
5 10
5 10
```

The answer is `20`: take items 2 and 3 (total weight `10 <= 10`, value `10 + 10 = 20`). Note that
item 1 has the best value-per-weight ratio (`13/6 ≈ 2.17`), yet including it is suboptimal here.

## Background

Two reflexes are immediately on the table, and both are disqualified by the constraints — which is
exactly what makes the regime interesting:

- **Weight-indexed dynamic programming.** The standard exact knapsack DP runs in `O(n * C)` time and
  `O(C)` memory by tabulating the best value for every capacity from `0` to `C`. Here `C` can be
  `10^18`, so a table indexed by capacity is impossible by many orders of magnitude. (A value-indexed
  DP indexed by total value — `O(n * sum(v))` — would need a table of size up to
  `40 * 10^9 = 4*10^10`, also far out of budget.)
- **Greedy by value-to-weight ratio.** Sort items by `v[i]/w[i]` descending and take greedily while
  they fit. This is `O(n log n)` and trivial, and it is *correct for the fractional knapsack*, which
  is why it is tempting. The open question is whether it is correct for the **0/1** knapsack, where
  items cannot be split.

What the constraints *do* leave open is that `n <= 40` is small enough to attack with subset
enumeration, provided one does not naively try all `2^40 ≈ 10^12` subsets.

## Evaluation settings

Judged on hidden tests covering: tiny capacities where almost nothing fits; capacities that exactly
match a weight threshold; capacity `0`; capacities larger than the total weight (so the answer is the
sum of all values); single items; all-items-too-heavy; values correlated and anti-correlated with
weights (to defeat ratio heuristics); and full-size instances with `n = 40` and weights/values near
`10^9` so that total values reach `~4*10^10` and require 64-bit arithmetic.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long C;
    if (!(cin >> n >> C)) return 0;
    vector<long long> w(n), v(n);
    for (int i = 0; i < n; i++) cin >> w[i] >> v[i];

    // TODO: compute the maximum total value of a subset with total weight <= C.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
