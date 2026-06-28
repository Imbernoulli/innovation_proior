# Rod cutting: maximum revenue from integer-length pieces

## Research question

You are given a rod of integer length `n` and a price list `p[1..n]`, where `p[k]` is the revenue you
earn for a single piece of length exactly `k`. You may cut the rod into any number of integer-length
pieces (including not cutting it at all, i.e. selling it whole as one length-`n` piece). Every cut is
free. Choose a set of cuts so that the **total revenue** — the sum of the prices of the resulting
pieces — is as large as possible, and output that maximum revenue.

Formally: maximize `p[c_1] + p[c_2] + ... + p[c_m]` over all compositions `c_1 + c_2 + ... + c_m = n`
with each `c_j >= 1`. There is always at least one way to cut (the whole rod), so the answer is well
defined for every `n >= 1`; for `n = 0` the rod is empty and the revenue is `0`.

This is the classic rod-cutting problem. The temptation is to pick pieces "greedily" by their
price-per-unit-length; whether that greedy choice is actually optimal under the global
length-must-sum-to-`n` constraint is exactly the question to settle before committing to an algorithm.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 5000`). Then follow `n` integers
  `p[1], p[2], ..., p[n]` (`0 <= p[k] <= 10^9`), whitespace-separated (spaces or newlines).
- Output (stdout): a single line with the maximum achievable total revenue.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 4` with prices `p = [1, 5, 8, 9]` the answer is `10` (two pieces of length 2,
earning `5 + 5`).

## Background

The cut points are a global choice: the lengths of all pieces must sum to exactly `n`, so a decision
about one piece changes what lengths remain available for the rest. Two families of approach are on
the table before committing to one:

- **Greedy by price-per-length.** Repeatedly cut off a piece of the length `k` whose ratio `p[k] / k`
  is largest among lengths that still fit in the remaining rod, then recurse on what is left. This is
  cheap and intuitive — "always take the most valuable material per unit" — and the open question is
  whether a locally best ratio can be safely committed to under the global summation constraint.
- **Dynamic programming over rod length.** Let `dp[L]` be the best revenue obtainable from a rod of
  length `L`. Build `dp[0], dp[1], ..., dp[n]` in order; the open question is the exact recurrence
  relating `dp[L]` to the prices and to shorter subproblems.

## Evaluation settings

Judged on hidden tests covering: `n = 0` (empty rod), `n = 1`, monotonically increasing price lists
(which tempt "never cut"), monotonically decreasing price lists (which tempt "cut into all length-1
pieces"), price lists with zeros and isolated spikes, lists engineered so the best price-per-length
piece is a trap, and large `n = 5000` with prices near `10^9` (so the total revenue exceeds the
32-bit integer range).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> p(n + 1, 0);     // p[k] = price of a piece of length k (1-indexed)
    for (int k = 1; k <= n; k++) cin >> p[k];

    // TODO: compute the maximum total revenue over all ways to cut the rod into
    //       integer-length pieces (selling it whole counts as "one cut").
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
