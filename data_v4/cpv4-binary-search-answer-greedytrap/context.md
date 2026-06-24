# Splitting a greenhouse row among watering robots (minimize the last finish time)

## Research question

A greenhouse has `n` plant beds in a single row, left to right. Bed `i` needs `a[i]` liters of water
(`a[i] >= 0`). You own `k` identical watering robots. The beds must be cut into exactly `k`
**contiguous** blocks (a robot may be assigned a block; with `k > n` some robots simply get nothing,
which we model by allowing empty blocks). Each robot waters its whole block back to back, so the
**time** a robot is busy equals the **sum of the liters** in its block. All robots start at once, so
the greenhouse is finished when the *slowest* robot finishes — i.e. the largest block sum.

Choose where to cut so that the **maximum block sum is as small as possible**, and output that
minimum possible finish time.

This is a min-max partition of a sequence into contiguous runs. The same shape appears in load
balancing, file-to-disk packing, and "paint the fence with k painters" scheduling, so getting the
contiguity, the lower bound, and the feasibility test exactly right is what matters.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `k`
  (`0 <= n <= 2*10^5`, `1 <= k <= 10^9`); then `n` non-negative integers `a[i]`
  (`0 <= a[i] <= 10^9`), whitespace-separated (they may span one or several lines, or none when
  `n = 0`).
- Output (stdout): a single line with the minimum possible value of the maximum block sum.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 7`, `k = 3`, `a = [2, 3, 5, 8, 1, 1, 4]` the answer is `10`
(cut into `[2,3,5] | [8] | [1,1,4]`, block sums `10, 8, 6`).

## Background

Two families of approach are on the table before committing to one:

- **Balance-by-average greedy.** The total water is `S = sum(a)`; with `k` robots the "fair share"
  is `S/k`. Walk left to right and cut each block as soon as its running sum reaches about `S/k` (or
  place the `k-1` cut points at the prefix sums closest to `S/k, 2S/k, ...`). This is `O(n)` and
  trivial to write. The open question is whether matching the average actually minimizes the
  *maximum* block — averages and maxima are not the same objective, and the blocks are forced to be
  contiguous.
- **Binary search on the answer.** Guess a candidate finish time `T`; ask "can the row be cut into at
  most `k` contiguous blocks each with sum `<= T`?" That feasibility question has an easy greedy
  answer (extend the current block while it fits, otherwise start a new one) and is **monotone** in
  `T`, so the smallest feasible `T` can be found by bisection. The open questions are the search
  bounds (what is the smallest conceivable `T`?) and getting the feasibility count exact.

## Evaluation settings

Judged on hidden tests covering: all-equal beds, beds with zeros, a single bed far larger than the
average (so a block must hold it), `k = 1` (one block = the whole sum), `k >= n` (every bed alone, so
the answer is the largest bed), `n = 0` (answer `0`), and large `n = 2*10^5` with values near `10^9`
(so the total can reach `2*10^14` and overflow a 32-bit integer).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long k;
    if (!(cin >> n >> k)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // TODO: minimize the maximum sum over k contiguous blocks covering a[0..n-1].
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
