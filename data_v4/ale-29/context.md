# Connected Region Selection (maximum-weight connected subgrid)

## Research question

You are given an `H x W` grid of integer cell weights `w[r][c]` (values may be
negative) and an integer budget `B`. Choose a set `S` of grid cells that is a
**single 4-connected component** (any two chosen cells are joined by a path of
chosen cells that step only up/down/left/right) and whose size is at most `B`,
so as to **maximize the sum of the weights of the chosen cells**. The empty set
is allowed, so the achievable score is never below `0`.

This is the grid instance of the maximum-weight connected subgraph problem
(NP-hard in general): you want to wrap a connected blob around the profitable
("positive") cells while routing *around* the toxic ("negative") ones, but the
connectivity constraint forbids splitting into disjoint profitable islands and
the budget caps how much area you can cover. The interesting instances are
"patchy": positive mass sits in a few shoals separated by negative moats, so the
optimal region is a non-convex blob with notches carved out of it -- exactly the
structure where a connectivity-aware local search beats one-shot greedy growth.

## Input / output contract

- Input (stdin):
  - Line 1: three integers `H W B`.
  - Next `H` lines: `W` integers each, the weight grid `w[r][c]`
    (`0 <= r < H`, `0 <= c < W`), whitespace-separated. Weights may be negative.
- Output (stdout):
  - Line 1: an integer `K`, the number of chosen cells (`0 <= K <= B`).
  - Next `K` lines: two integers `r c` each, the chosen cells (`0`-indexed).
  - `K = 0` (and an otherwise empty output) denotes the empty region.
- Time limit: about 2 seconds per instance. Memory: 256 MB.

The instances used here are fixed at `H = W = 60`, `B = 900` (a region of up to
a quarter of the grid).

## Background

Two reference approaches frame the problem:

- **Best single cell.** Output the one cell with the largest weight (or the empty
  region if all weights are non-positive). A trivial floor; ignores connectivity
  entirely because a single cell is always connected.
- **Greedy best-first region growth.** Seed at the globally best cell, then
  repeatedly add the frontier cell (a non-chosen cell 4-adjacent to the current
  region) with the largest weight, while that weight is positive and the budget
  allows. Growth bakes connectivity in -- every added cell touches the region --
  but it is *irreversible*: it can never shed a cell it regrets, never carve a
  notch around a negative pocket it grew over, and never spend budget to bridge
  across a thin negative isthmus to reach a richer shoal. This greedy is the
  normalization reference.

The established stronger method for max-weight connected subgraph under a size
budget is **local search that can both add and remove cells**, with the catch
that a removal must not disconnect the region. The non-obvious lever is to make
that connectivity guard *local and O(1)* instead of re-running a global
connectivity check after every candidate move.

## Evaluation settings

The scorer reads the instance and a candidate solution and computes:

- **Feasibility.** The solution lists `K` cells. It is feasible iff
  (1) `0 <= K <= B`; (2) every listed cell is in range; (3) the cells are
  pairwise distinct; and (4) the listed cells form a single 4-connected
  component (`K = 0`, the empty region, is trivially feasible).
- **Score.** If feasible, the raw score is the **sum of the weights of the
  chosen cells**, reported as `max(0, sum)` -- the empty region with sum `0` is
  always available, so a feasible output never scores below `0`. If
  **infeasible** (out of range, duplicate cell, over budget, or more than one
  connected component), the score is **floored to `0`** -- the feasibility-to-0
  rule that makes an invalid net worthless.

We report the mean raw score over a fixed seed set (seeds `1..20`), each run on
the same instances so the numbers are directly comparable, and normalize against
the greedy best-first growth baseline.

**How instances are generated.** A generator takes an integer seed and builds a
smooth low-frequency field as a sum of 10-16 random 2D Gaussian bumps with mixed
signs (some attractive, some repulsive), adds per-cell Gaussian noise, and
applies a global downward shift so a healthy fraction of cells are negative
("toxic"). Rounding to integers yields a weight grid with a few profitable shoals
separated by negative regions -- patchy by construction.

## Code framework

A single self-contained C++17 program that reads the instance and writes a
feasible region. The scaffold below establishes the I/O and a guaranteed-feasible
fallback (the best single cell); the heuristic replaces the `// TODO`.

```cpp
#include <bits/stdc++.h>
using namespace std;

int H, W, B;
vector<int> wgt;            // weight grid, row-major, size H*W
inline int ID(int r, int c) { return r * W + c; }

int main() {
    if (scanf("%d %d %d", &H, &W, &B) != 3) return 0;
    int N = H * W;
    wgt.resize(N);
    for (int i = 0; i < N; i++) scanf("%d", &wgt[i]);

    // A connected region of cells (4-adjacency), size <= B, maximizing the sum
    // of chosen weights. The empty region (output "0") is always feasible.
    vector<int> cells;

    // TODO heuristic: build a strong connected region into `cells`.
    // Feasible fallback baked in below: pick the single best positive cell.
    if (cells.empty() && B > 0) {
        int best = -1;
        for (int i = 0; i < N; i++)
            if (wgt[i] > 0 && (best < 0 || wgt[i] > wgt[best])) best = i;
        if (best >= 0) cells.push_back(best);  // else leave empty (score 0)
    }

    printf("%d\n", (int)cells.size());
    for (int id : cells) printf("%d %d\n", id / W, id % W);
    return 0;
}
```
