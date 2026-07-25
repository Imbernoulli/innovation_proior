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

Because you are allowed to perform zero fusions, the answer is always at least `0`. This is the
one-dimensional core of a family of "optimal bracketing / merge-order" problems (matrix-chain,
stone-merging, polygon triangulation), where the order in which adjacent pieces are combined changes
the total cost.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 400`); then `n` integers `c[i]`
  (`-10^4 <= c[i] <= 10^4`), whitespace-separated.
- Output (stdout): a single line with the maximum total fusion reward.
- Time limit: 1 second. Memory: 256 MB.

Example: for `c = [3, 1, 5, 8]` the answer is `95`. One optimal schedule fuses `3 & 1` (reward
`3`), then `4 & 5` (reward `20`), then `9 & 8` (reward `72`): `3 + 20 + 72 = 95`. Every crystal ends
up in one cluster, but a *different* fusion order yields less, so the order matters.

## Evaluation settings

Judged on hidden tests covering: all-positive rows, rows mixing negatives and zeros, the empty row
(`n = 0`), a single crystal (`n = 1`), all-negative rows, rows of all zeros, rows where the optimum
is to fuse only some contiguous blocks and leave the rest alone, and large `n = 400` with charges
near `10^4`.

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
