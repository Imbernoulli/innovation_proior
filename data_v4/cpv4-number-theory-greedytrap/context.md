# Fewest equal-order pulses to deposit a target energy

## Research question

A calibration rig fires laser pulses into a target. The hardware is locked to a single *order*
`k`, and every pulse it can emit carries an energy that is a perfect `k`-th power of a positive
integer: the admissible pulse energies are `1^k, 2^k, 3^k, ...` (so `1, 4, 9, 16, ...` when
`k = 2`; `1, 8, 27, 64, ...` when `k = 3`; and so on). You may fire any pulse as many times as you
like, in any order, and the deposited energies simply add up.

You are given the order `k` and a target energy `n`. Output the **minimum number of pulses** needed
so that the deposited energies sum to **exactly** `n`. Because `1 = 1^k` is always an admissible
pulse for every `k`, every target is reachable, so an answer always exists.

This is the "minimum number of perfect `k`-th powers summing to `n`" problem — a sums-of-powers
question in additive number theory (the `k = 2` case is the classical "sum of fewest squares"). The
point of interest is that the most natural one-line strategy for it is subtly and provably wrong.

## Input / output contract

- Input (stdin): a single line with two integers `k` and `n`, separated by whitespace, with
  `2 <= k <= 5` and `0 <= n <= 10^6`.
- Output (stdout): a single line with the minimum number of admissible pulses whose energies sum to
  exactly `n`. For `n = 0` the answer is `0` (fire nothing).
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `k = 2`, `n = 12` the answer is `3` (use `4 + 4 + 4`).

## Background

The constraint "each summand is a perfect `k`-th power" makes this a constrained representation
problem, and two families of approach are on the table before committing to one:

- **Greedy by largest power.** Repeatedly subtract the largest admissible power that does not exceed
  the remaining target, counting one pulse each time, until the remainder hits `0`. It is
  `O(answer)` and three lines to write; the open question is whether always grabbing the largest
  fitting power is actually optimal, given that an overshoot in "coverage" early can force many tiny
  `1`s later.
- **Shortest-representation dynamic programming.** Treat the values `0..n` as states and define
  `dp[v]` = the fewest powers summing to `v`; then `dp[v] = 1 + min over admissible powers p <= v of
  dp[v - p]`. This is `O(n * P)` where `P` is the count of admissible powers up to `n`; the open
  questions are the exact recurrence, the base case, and the data types used to enumerate the powers
  without overflow.

## Evaluation settings

Judged on hidden tests covering: `k = 2` targets where greedy famously fails (`12`, `18`, `32`,
`128`); other orders `k = 3, 4, 5`; the corners `n = 0` and `n = 1`; targets that are themselves
exact `k`-th powers (answer `1`); targets near the upper bound `n = 10^6` for each `k` (so the DP
table and the power-enumeration loop are both stressed); and a target like `n` with no small
representation that forces a long chain of `1`s for large `k`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    long long k, n;
    if (!(cin >> k >> n)) return 0;

    // TODO: output the minimum number of perfect k-th powers (1^k, 2^k, ...)
    //       whose sum is exactly n. (n == 0 -> 0.)
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
