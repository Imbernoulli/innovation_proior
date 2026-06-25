# Sequencing a render farm to minimize total weighted wait

## Research question

A single render node processes `n` scenes back to back, one at a time, starting at clock `0`. Scene
`i` needs `p[i]` seconds of compute and has an **impatience weight** `w[i]` (the cost charged for
every second the client who owns scene `i` is kept waiting). If you fix an order and scene `i`
finishes at clock time `C[i]` (the cumulative compute time of itself and everything scheduled before
it), that scene contributes `w[i] * C[i]` to the bill. You choose the processing order of all `n`
scenes; you must run every scene, with no gaps and no preemption.

Output the **minimum** achievable total bill `sum over i of w[i] * C[i]`.

This is the classic single-machine *total weighted completion time* objective. It is the engine room
of a lot of scheduling: the right order is decided by a pairwise *exchange* argument, and the whole
problem hinges on comparing two jobs *exactly* — getting that comparison and the final accumulation
arithmetic right is the entire difficulty here.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 10^5`). Then `n` lines (or just whitespace-
  separated pairs) follow, each `p[i] w[i]` with `1 <= p[i] <= 10^9` and `1 <= w[i] <= 10^9`.
- Output (stdout): a single line with the minimum total weighted completion time.
- Time limit: 1 second. Memory: 256 MB.

Example: for the three scenes `(p,w) = (3,1), (1,2), (2,5)` the answer is `22`.

## Background

The decision is purely about *order*, and order problems of this shape are governed by an adjacent-
exchange (Smith's-rule) argument: if two neighbouring jobs are in the wrong relative order, swapping
them lowers the bill, so the optimum is the order in which no adjacent swap helps. Two routes are on
the table before committing:

- **Greedy by a single key.** Sort the scenes by some per-scene key and run them in that order. The
  open questions are *which* key (raw `p`? raw `w`? the ratio `p/w`?) and how to compare two keys
  *without* losing information — because the values are large, a floating-point ratio is suspect.
- **Search / DP over subsets.** Treat it as "pick the next scene from the remaining set", which is a
  bitmask DP. It is obviously correct but `O(2^n * n)`, hopeless past `n ~ 20`; it is useful only as
  an oracle to check a greedy against, not as the shipped solution.

## Evaluation settings

Judged on hidden tests covering: tiny hand-checkable instances; `n = 0` and `n = 1`; many scenes
with values near `10^9` so the running clock and the objective both grow far past 32-bit and even
past 64-bit range; and **adversarial near-tie ratios**, i.e. pairs of scenes whose `p/w` ratios are
so close that a `double` comparison cannot tell them apart even though their exact order changes the
bill. A solution that sorts on `double` ratios, or that accumulates the bill in a 64-bit integer,
is expected to fail some of these.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> p(n), w(n);
    for (int i = 0; i < n; i++) cin >> p[i] >> w[i];

    // TODO: order the scenes to minimize sum of w[i] * (completion time of i),
    // comparing scenes exactly and accumulating the bill without overflow.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
