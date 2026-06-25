# Two cranes clearing a container row on one shared fuel tank

## Research question

A single row of `n` shipping containers waits on a dock, left to right; container `i` weighs
`w[i]` (a positive integer). Two cranes clear the row in one shift:

- the **left crane** only ever lifts the *leftmost* remaining container, so it removes some
  **prefix** `w[0], w[1], ..., w[i-1]` (the first `i` containers, possibly `i = 0`);
- the **right crane** only ever lifts the *rightmost* remaining container, so it removes some
  **suffix** `w[n-1], w[n-2], ..., w[n-j]` (the last `j` containers, possibly `j = 0`).

The two cranes draw fuel from **one shared tank** holding `B` units, and lifting a container
burns fuel equal to its weight. So the total weight the cranes remove must satisfy
`(sum of the prefix) + (sum of the suffix) <= B`. They also cannot fight over the same container,
so the prefix and suffix must not overlap: `i + j <= n`.

Maximize the **number of containers removed**, `i + j`. Output that maximum.

This is a prefix/suffix selection under one budget — the structural core of many "consume from both
ends until a shared resource runs out" tasks (two-sided sliding windows, dock/queue clearing,
bidirectional scanning). Getting the shared-budget interaction and the no-overlap corner exactly
right is the whole game.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `B`
  (`0 <= n <= 2*10^5`, `0 <= B <= 10^18`). The second line has `n` integers `w[i]`
  (`1 <= w[i] <= 10^9`), whitespace-separated. When `n = 0` the second line is empty/absent.
- Output (stdout): a single line with the maximum number of containers that can be removed.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 6`, `B = 11`, `w = [1, 7, 7, 8, 2, 8]` the answer is `3` — the left crane takes
the single container `w[0] = 1` (cost 1) and the right crane takes the last two, `w[5] = 8` and
`w[4] = 2` (cost 10), for a total cost of `11 <= B` and `3` containers removed.

## Background

The constraint that fuel is *shared* couples the two ends: every unit the left crane burns is a unit
the right crane cannot. Two families of approach are on the table before committing to one.

- **Greedy from the cheaper end.** Keep a pointer at each end; repeatedly lift whichever end's next
  container is lighter (it costs the least fuel right now), stopping when neither end's container
  fits the remaining fuel. This is `O(n)` and very natural — grab the cheapest thing available. The
  open question is whether always burning fuel on the locally cheapest container actually maximizes
  the *count*, given that the two ends compete for one tank.
- **Two-pointer sweep over prefix lengths.** Consider every affordable prefix length `i` in order,
  and for each one take the largest suffix the leftover fuel and the no-overlap bound allow. The open
  question is the exact monotonic relationship that lets the suffix pointer move only one direction,
  keeping the sweep `O(n)`, and the precise placement of the overlap cap.

## Evaluation settings

Judged on hidden tests covering: all-equal weights, heavy/light alternating rows that bait the
cheaper-end greedy, `B = 0` (remove nothing), `B` at least the total weight (remove everything,
testing the overlap cap), `n = 0` and `n = 1`, rows where the optimum is entirely on one side, and
large `n = 2*10^5` with weights near `10^9` and `B` near `10^18` (so prefix/suffix sums and `B`
both need 64-bit arithmetic).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long B;
    if (!(cin >> n >> B)) return 0;
    vector<long long> w(n);
    for (auto &x : w) cin >> x;

    // TODO: max number of containers removable as a non-overlapping prefix+suffix
    //       whose combined weight is at most the shared budget B.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
