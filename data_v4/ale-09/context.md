# Dynamic Bin Packing with Rebalancing

## Research question

Items arrive and depart over time; each item must live in exactly one fixed-capacity bin for its
whole lifetime, and a bin's used capacity at any instant is the total size of the items that are alive
in it at that instant. You are given all items up front (offline) and must produce, for every item,
the bin it is placed into, so that **no bin ever exceeds its capacity** and the **number of distinct
bins used is as small as possible**.

Because items *depart*, a bin that is full now can become reusable later — so this is not ordinary bin
packing but **temporal / dynamic bin packing** (a.k.a. interval bin packing or dynamic storage
allocation). It is NP-hard, there is no known efficient optimum, and the score is continuous: how close
you get to the unavoidable lower bound (the peak simultaneous load divided by capacity). The lever is a
strong construction plus a rebalancing local search that repeatedly tries to free whole bins.

## Input / output contract

- Input (stdin):
  - First line: `N C` — the number of items (`1 <= N <= 1200`) and the per-bin capacity
    (`1 <= C`). (`N` may be `0`, in which case there is nothing to output.)
  - Then `N` lines, the `i`-th being `a_i d_i s_i`: item `i` is **alive on the half-open interval
    `[a_i, d_i)`** (`0 <= a_i < d_i`) and consumes integer size `s_i` (`1 <= s_i <= C`) while alive.
- Output (stdout): `N` lines; line `i` is the bin index `b_i` (a **non-negative integer**) that item
  `i` is assigned to. Bin indices need not be contiguous, but using fewer *distinct* indices is better.
- Time limit: about 2 seconds. Memory: 256 MB.

Example: `N=3, C=10`, items `(0,5,6) (0,5,6) (0,5,6)` all overlap and each needs more than half the
capacity, so no two share a bin — the best answer uses 3 bins (e.g. `0`, `1`, `2`).

## Background

Two ingredients are on the table before committing to a method:

- **Online best-fit construction.** Process items in arrival order and drop each into the *tightest*
  bin it still fits (largest peak load over the item's lifetime), opening a new bin only when none fits.
  Best-fit keeps bins dense, which is exactly what later makes whole bins emptyable. The open question
  is how to test "fits over `[a_i, d_i)`" cheaply, because a naive recheck of a bin's whole schedule is
  expensive when there are hundreds of bins and thousands of items.
- **Rebalancing local search.** A construction alone gets stuck; the number of bins drops only if some
  bin can be fully *emptied* by relocating its items elsewhere. The open question is the neighborhood:
  blindly moving single items rarely frees a bin, and re-evaluating capacity from scratch after every
  move is far too slow inside a time budget.

The non-obvious design choices are (1) an **incremental fill representation** — a per-bin time-profile
array so that placing, removing, or testing an item touches only the cells in its lifetime window, and
(2) a **"repack the fullest/lightest few bins"** move that tries to *empty a target bin entirely*
(ruin-and-recreate / LNS), which is what actually reduces the bin count, rather than aimless single-item
swaps.

## Evaluation settings

The solution is scored by a deterministic local scorer (`verify/score.py`):

- **Feasibility (floor to 0).** The output must be exactly `N` non-negative integers. For every bin,
  a sweep line over its items' `(+s at a_i, -s at d_i)` events must never let the alive load exceed `C`
  (it suffices to check at arrival instants, since a bin's load only rises at arrivals). If the output
  is malformed (wrong count, non-integer, negative index, missing file) **or any bin overflows at any
  instant**, the solution is **infeasible and scores 0**.
- **Objective.** Let `K` be the number of *distinct* bin indices the solution actually uses. Let `B`
  be the number of bins used by the deterministic **first-fit-by-arrival baseline**, which the scorer
  recomputes itself (items sorted by `(arrival, departure, size, index)`, each placed into the
  lowest-indexed bin where it fits at its arrival instant, opening a new bin otherwise). For a feasible
  solution, `SCORE = round(1_000_000 * B / K)`. The baseline scores exactly `1_000_000`; using fewer
  bins scores strictly more; using more scores less but stays positive. Higher is better.

**Instances** (`verify/gen.py`, parameter: integer `seed`) are generated deterministically: `N` in
`[400, 1200]`, capacity `C` in `[20, 60]`, a time horizon `T` scaling with `N`, and a mixture of
long-lived "background" items (large size, span much of the horizon) and many short-lived "burst" items
(small-to-medium size). This regime makes the peak simultaneous load comfortably exceed `C` — forcing
many bins — while leaving genuine reassignment slack that the first-fit baseline wastes.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, C;
    if (scanf("%d %d", &N, &C) != 2) return 0;
    if (N <= 0) return 0;
    vector<int> A(N), D(N), S(N);
    for (int i = 0; i < N; ++i) scanf("%d %d %d", &A[i], &D[i], &S[i]);

    // TODO: assign every item to a bin so that no bin's alive load ever exceeds C,
    // minimizing the number of distinct bins used. Start from a feasible best-fit
    // construction (always valid), then rebalance to free whole bins.
    vector<int> bin(N, 0);

    // print one bin index per item, in item order
    string out;
    for (int i = 0; i < N; ++i) { out += to_string(bin[i]); out += '\n'; }
    fwrite(out.data(), 1, out.size(), stdout);
    return 0;
}
```
