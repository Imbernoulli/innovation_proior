# Online Bin Assignment (sequential, partial information)

## Research question

A stream of items arrives one at a time. There are `K` bins with fixed integer
capacities `c_1, ..., c_K`. Each item has a size `s` and a value `v`. When an
item arrives you must **immediately and irrevocably** assign it to one of the
`K` bins (if it still fits) or **drop** it; you may not see any future item
before committing the current one, and you may never move an item once placed.
The total size of items placed in a bin may never exceed that bin's capacity.

The objective is to **maximize the total value of all placed items**. This is
an online, capacitated, value-maximization assignment problem (a multi-knapsack
online problem): the contest is *which* items deserve scarce capacity, decided
under partial information about the rest of the stream.

## Input / output contract

This is an **interactive** problem. The judge (the scorer) reveals the instance
incrementally and reads one decision per item before revealing the next, so the
solver physically cannot look ahead.

Interaction protocol (stdin / stdout of the solver):

```
read:  K N                      # number of bins, number of items
read:  c_1 c_2 ... c_K          # K integer bin capacities
repeat N times:
    read:  s v                  # the next item's size and value
    write: b                    # b in 1..K to place the item in bin b,
                                #   or b == 0 to DROP it
    flush                       # MUST flush before the next item is sent
```

The solver must emit **exactly one** integer per item, in arrival order, and
flush after each (the judge will not send item `i+1` until it has read the
decision for item `i`). Dropping (`0`) is always legal.

Constraints: `1 <= K <= 12`; `2000 <= N <= 4000`; `1 <= c_b`; `1 <= s, v`.
Time budget: 2 seconds of solver CPU over the whole stream (the reference
solver finishes in well under 0.1 s). Memory: 256 MB.

## Background

Two reference strategies frame the problem:

- **First bin that fits (trivial baseline).** For each item, place it in the
  first bin with enough remaining capacity, else drop. Always feasible, `O(K)`
  per item. Its weakness: it spends capacity on whatever arrives early,
  regardless of value, so once bins fill it has no way to have saved room for
  the high-value items that arrive later.
- **Adaptive value-density threshold (the strong method).** Treat each item by
  its value-density `d = v / s` and accept an item only if its density clears an
  adaptive cutoff estimated from the empirical density distribution seen so far,
  reserving scarce capacity for the densest items. This is the secretary /
  online-knapsack threshold family: under partial information the right cutoff
  is data-driven, not a fixed constant, because the value distribution is
  non-stationary and the contention (total demand vs total capacity) varies by
  instance.

## Evaluation settings

The scorer is an **offline-simulated online judge**. It reads an instance file,
launches the solver as a subprocess, sends the header, then for each item sends
`s v`, flushes, and reads back exactly one token (the bin id or `0`). It tracks
remaining capacity per bin and the running total of placed value.

Scoring rule, with the feasibility floor:

- A decision `b` in `1..K` places the item into bin `b`; `b == 0` drops it.
- If **any** placement would overflow a bin (placed-size exceeds capacity), or
  the output is malformed (out-of-range bin id, a non-integer, the wrong number
  of tokens), or the solver crashes or times out, the score is **floored to 0**.
- Otherwise the **raw score is the total value of all placed items**.

The reported metric on the seed set is the raw placed value. For context it is
also normalized against the trivial "first bin that fits" baseline computed by
the same scorer (`score.py INSTANCE --baseline`), under identical feasibility
rules so the numbers are directly comparable.

How instances are made (`gen.py SEED`): `K` in `[4, 12]` bins with capacities in
`[800, 1600]`; `N` in `[2000, 4000]` items. Item sizes are in `[1, 120]` (small
relative to a bin, so packing is about value selection, not bin-packing tetris).
Values come from a **non-stationary** distribution: a slowly drifting baseline
mean plus occasional high-value spikes, mildly correlated with size and heavily
noised. Total item size far exceeds total capacity (typically 10x-50x), so bins
fill and the only question that matters is which value you keep -- exactly the
regime where an adaptive acceptance threshold pays off. The generator, scorer,
and seed set are frozen.

## Code framework

A single self-contained C++17 program that speaks the interactive protocol.
Below is a pre-method scaffold: it reads the header, then for each item reads
`s v`, makes a placeholder decision, prints one feasible bin id (or `0`), and
flushes. The `// TODO` is where the adaptive-threshold policy goes.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    // do NOT untie cin/cout: we must flush after every decision.

    int K; long long N;
    if (!(cin >> K >> N)) return 0;
    vector<long long> cap(K), rem(K);
    for (int i = 0; i < K; i++) { cin >> cap[i]; rem[i] = cap[i]; }

    for (long long it = 0; it < N; it++) {
        long long s, v;
        if (!(cin >> s >> v)) break;

        int choice = 0;  // default: drop (always feasible)

        // TODO: adaptive value-density threshold under partial info.
        //   Decide a bin b in 1..K (only if rem[b-1] >= s) or 0 to drop.
        //   Must never overflow a bin: only return b with rem[b-1] >= s.

        if (choice >= 1) rem[choice - 1] -= s;  // commit
        cout << choice << "\n";
        cout.flush();                            // interactive: flush each line
    }
    return 0;
}
```
