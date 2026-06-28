# Rectangle Strip Packing (minimize used strip height)

## Research question

You are given a strip of fixed integer width `W` and unbounded height, and `N`
axis-aligned rectangles (no rotation), the `i`-th of size `w_i x h_i` with
`1 <= w_i <= W`. Place every rectangle, without overlap, so that each lies fully
inside the strip horizontally (`0 <= x_i` and `x_i + w_i <= W`) and on or above
the floor (`y_i >= 0`). Minimize the **used height**

```
H = max_i (y_i + h_i),
```

the highest top edge over all placements. This is the classic **two-dimensional
strip packing problem** (2D-SPP), `NP`-hard, with no closed-form optimum; it is
judged on a continuous score, so the task is to pack as densely as possible
under a wall-clock budget rather than to find a provably optimal packing.

## Input / output contract

- Input (stdin): the first line is `W N` (`W = 1000`, `30 <= N <= 200`). Then `N`
  lines follow, the `i`-th being `w_i h_i` with `1 <= w_i <= W`, `1 <= h_i`.
- Output (stdout): exactly `N` lines, the `i`-th being `x_i y_i`, the integer
  coordinates of the **bottom-left corner** of rectangle `i`, in the **same
  order** as the input. Rectangle `i` then occupies `[x_i, x_i+w_i) x
  [y_i, y_i+h_i)`.
- Time limit: about 2 seconds per instance. Memory: 256 MB.

## Background

Strip packing is one of the canonical NP-hard cutting-and-packing problems. Two
ingredients define essentially every strong heuristic for it:

- **A placement rule that turns an ordering into a packing.** The standard rule
  is **Bottom-Left-Fill (BLF)**: process the rectangles in a given order and put
  each one as low as it will go, breaking ties to the left. BLF of *any*
  permutation is always a legal, non-overlapping packing, which makes it a safe
  decoder. Implemented naively (re-scanning all placed rectangles for every
  candidate position) it is slow; implemented over a **skyline** (the upper
  contour of what is already packed) each placement only inspects the few
  contour segments it spans.

- **A search over orderings.** Because BLF height depends only on the
  permutation, the optimization collapses to "find the insertion order whose BLF
  packing is lowest." Sorting heuristics (decreasing height / decreasing width /
  decreasing area) give a strong first packing; a metaheuristic over the
  permutation — here **simulated annealing** with swap / insert / reverse moves —
  improves on it. This BLF-decode + permutation-search pairing is the
  best-known practical family for one-strip 2D-SPP.

## Evaluation settings

Let `LB = max( ceil( (sum_i w_i*h_i) / W ),  max_i h_i )` be a lower bound on
any feasible height (total area cannot fit below `area/W`, and no packing can be
shorter than the tallest single rectangle). For a submitted solution the scorer:

1. parses exactly `N` integer pairs (fewer/garbled tokens -> **infeasible**);
2. checks `0 <= x_i`, `x_i + w_i <= W`, `y_i >= 0` for every `i` (out of the
   strip -> **infeasible**);
3. checks that no two rectangle interiors overlap; edges/corners may touch
   (any interior overlap -> **infeasible**);
4. if feasible, sets `H = max_i (y_i + h_i)` and returns

```
score = 100 * LB / H        (H > 0),
score = 100                 (H == 0, i.e. N == 0).
```

Any infeasibility floors the score to **0**. The score lies in `(0, 100]`:
higher is better, `100` is the (generally unreachable) area/height bound, and a
trivial shelf packer scores around 45-60 on these instances. Instances are made
by `gen.py SEED`: `W = 1000`, `N` drawn from `[30, 200]`, and each rectangle is
one of three shapes (wide-and-short, tall-and-narrow, or general) so that no
single shelf orientation packs well; all randomness is seeded for
reproducibility.

## Code framework

A single self-contained C++17 program that reads the instance from stdin and
writes a feasible packing to stdout. The scaffold below already emits a valid
(if poor) packing — every rectangle on its own row — so the contract is
satisfied before any optimization is added.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int W, N;
    if (scanf("%d %d", &W, &N) != 2) return 0;
    vector<int> w(N), h(N);
    for (int i = 0; i < N; ++i) scanf("%d %d", &w[i], &h[i]);

    vector<int> x(N), y(N);
    // Feasible baseline: stack every rectangle in its own full-width row.
    int yc = 0;
    for (int i = 0; i < N; ++i) { x[i] = 0; y[i] = yc; yc += h[i]; }

    // TODO heuristic: BLF decode over a skyline + simulated annealing on the
    // insertion order to minimize H = max_i (y[i] + h[i]).

    for (int i = 0; i < N; ++i) printf("%d %d\n", x[i], y[i]);
    return 0;
}
```
