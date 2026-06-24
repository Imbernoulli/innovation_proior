# Crystal annealing: maximum fusion reward

## Research question

A row of `n` crystals carries integer charges `c[0..n-1]` (each charge may be negative, zero, or
positive). You operate an annealing furnace that performs **fusions**. A single fusion takes two
**currently adjacent** clusters, the left one carrying charge `L` and the right one carrying charge
`R`, and welds them into one cluster of charge `L + R`. The energy released by that fusion — the
**reward** — is the product `L * R` (which can be negative, zero, or positive). Initially every
crystal is its own cluster. You may perform any sequence of fusions, in any order, and you may
**stop whenever you like** (including never fusing anything). Output the maximum total reward
achievable, summed over all the fusions you perform.

Because you are allowed to perform zero fusions, the answer is always at least `0`. But note that
"do nothing" is not automatically optimal even when every charge is negative: fusing two negative
clusters releases a *positive* reward `L * R > 0`. So the all-negative input is a genuine trap — the
answer there is usually large and positive, not `0`. This is the one-dimensional core of a family of
"optimal bracketing / merge-order" problems (matrix-chain, stone-merging, polygon triangulation),
where the order in which adjacent pieces are combined changes the total cost, and where sign handling
on the reward is the subtle part.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 400`); then `n` integers `c[i]`
  (`-10^4 <= c[i] <= 10^4`), whitespace-separated.
- Output (stdout): a single line with the maximum total fusion reward.
- Time limit: 1 second. Memory: 256 MB.

Example: for `c = [3, 1, 5, 8]` the answer is `95`. One optimal schedule fuses `3 & 1` (reward
`3`), then `4 & 5` (reward `20`), then `9 & 8` (reward `72`): `3 + 20 + 72 = 95`. Every crystal ends
up in one cluster, but a *different* fusion order yields less, so the order matters.

## Background

The reward of fusing two clusters depends only on the two charges at the moment of welding, and the
charge of a cluster is the sum of the original charges it contains. So the whole process is governed
by where you place the "last weld" inside each maximal cluster you build. Two families of approach
are on the table before committing to one:

- **Greedy / local rules.** Always fuse the currently most rewarding adjacent pair, or always fuse
  left-to-right, or fuse all same-sign runs. Each is `O(n log n)`-ish and easy to write; the open
  question is whether a locally best weld is ever globally suboptimal under the reordering freedom.
- **Interval dynamic programming.** For each contiguous range, compute the best reward to fuse that
  range into a single cluster by trying every position of the *last* weld, then layer a partition DP
  on top to decide which ranges to fully fuse and which crystals to leave alone. This is `O(n^3)`;
  the open questions are the exact recurrence, the base case for a length-1 range, and how the
  "stop / leave alone" option enters so that the answer can be `0`.

## Evaluation settings

Judged on hidden tests covering: all-positive rows, rows mixing negatives and zeros, the empty row
(`n = 0`), a single crystal (`n = 1`, answer `0`), all-negative rows (answer is positive, *not* `0`,
because negative times negative is positive), rows of all zeros (answer `0`), rows where the optimum
is to fuse only some contiguous blocks and leave the rest alone, and large `n = 400` with charges
near `10^4` (so the running total exceeds a 32-bit integer and the `O(n^3)` DP must still fit in the
time limit).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> c(n);
    for (auto &x : c) cin >> x;

    // TODO: compute the maximum total fusion reward.
    //   - reward of welding clusters with charges L and R is L * R,
    //   - the empty schedule (no fusions) is allowed, so the answer is >= 0.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
