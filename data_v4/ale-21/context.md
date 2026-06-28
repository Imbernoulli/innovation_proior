# Guillotine Cutting Stock

## Research question

You run a cutting shop with an unlimited supply of identical rectangular sheets of stock, each `W`
units wide and `H` units tall. A customer order lists `n` requested rectangles; rectangle `i` has
width `w_i` and height `h_i`. You must place requested rectangles onto sheets so that:

- every placement is axis-aligned and may be rotated 90 degrees (swap width and height);
- placements on a sheet do not overlap and stay inside the sheet; and
- the layout on each sheet is realizable by **guillotine cuts** only — a guillotine cut runs
  edge-to-edge across the current piece (full width or full height), and you cut recursively.
  Equivalently, the set of rectangles on a sheet must be separable by a sequence of full-length
  cuts, each cut not crossing the interior of any placed rectangle.

You want to **use few sheets and waste little area**. Leaving a requested rectangle unplaced is
allowed but penalized. This is the classic two-dimensional guillotine cutting-stock problem; it is
NP-hard, has no known closed-form optimum, and is judged by a continuous score.

## Input / output contract

- Input (stdin): the first line is `n W H` (`1 <= n <= 90`, `100 <= W, H <= 300`). Then `n` lines
  follow, the `i`-th being `w_i h_i` with `1 <= w_i <= W'` and `1 <= h_i <= H'` for the larger sheet
  dimension (every rectangle fits the sheet in at least one orientation).
- Output (stdout): the first line is `m`, the number of placed rectangles (`0 <= m <= n`). Then `m`
  lines follow, each `idx sheet x y rot`:
  - `idx` — the 0-based index of the requested rectangle (each index used at most once);
  - `sheet` — a sheet id `>= 0` (sheet ids you actually use define the opened sheets);
  - `x y` — integer coordinates of the rectangle's bottom-left corner on that sheet;
  - `rot` — `0` to use `w_i x h_i`, `1` to use the rotated `h_i x w_i`.
  Any requested rectangle not listed is left unplaced. The order of the lines is free.
- Time limit: 2 seconds. Memory: 256 MB.

## Background

Two families of approach are on the table before committing to one.

- **Shelf / next-fit construction.** Sort or stream rectangles into horizontal "shelves" (bands): fill
  a shelf left to right, open a new shelf above when the next rectangle does not fit, open a new sheet
  when no shelf fits. Shelf layouts are trivially guillotine-legal (cut between bands, then between
  rectangles in a band). It is `O(n)` and easy, but it leaves the vertical slack inside each shelf
  unused, so it opens far more sheets than necessary — sheets are the dominant cost.
- **Free-rectangle (guillotine k-d tree) construction with search.** Represent each sheet as a pool of
  free rectangles. Placing a rectangle consumes a free rectangle and **guillotine-splits** the
  leftover L-shape into two free rectangles. Because every step is "drop into a free rectangle + one
  guillotine cut", feasibility is automatic. The open question is the *insertion order and rotation*,
  which a metaheuristic can search by replaying the construction.

The non-obvious lever is to **bake feasibility into the construction** so the search never has to
repair an illegal layout — then optimize purely over the insertion sequence.

## Evaluation settings

A deterministic local scorer (`verify/score.py`) reads the instance and a candidate solution and prints
an integer score; higher is better.

- **Feasibility (any violation floors the score to 0):** the output parses as `m` followed by exactly
  `m` well-formed lines; every `idx` is in range and used at most once; every `rot` is `0` or `1`; each
  placed rectangle lies fully inside its sheet; and within every sheet the placed rectangles are
  pairwise non-overlapping **and** guillotine-separable (checked by a recursive full-cut search).
- **Cost of a feasible solution (lower is better).** Let `S` be the number of distinct sheets that hold
  at least one rectangle, `placed` the total placed area, `unplaced` the total area of omitted
  rectangles, and `P = 3` the unplaced penalty. Then
  `cost = S * W * H - placed + P * unplaced`.
  Every opened sheet costs its full area, recovered by what we actually place on it, plus a penalty per
  unit of unplaced requested area.
- **Score (normalized, higher better).** The scorer recomputes the cost of a deterministic **shelf
  next-fit baseline** (`baseline_cost`) and reports `score = round(1_000_000 * baseline_cost /
  max(1, solver_cost))`. The baseline scores about `1_000_000`; a lower-cost packing scores more.
  Infeasible output scores `0`.
- **Instances.** `verify/gen.py <seed>` builds an instance by recursively guillotine-splitting a few
  `W x H` template sheets into pieces (so a dense packing genuinely exists), then jittering each
  piece's dimensions by a few units and randomly rotating it. `n` is drawn in `[40, 90]` and `W, H` in
  a few-hundred range. The jitter and rotation make the optimum non-trivial while keeping the instance
  packable, which is exactly the regime where ordering and best-fit decisions matter.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, W, H;
    if (scanf("%d %d %d", &n, &W, &H) != 3) return 0;
    vector<int> w(n), h(n);
    for (int i = 0; i < n; i++) scanf("%d %d", &w[i], &h[i]);

    // TODO: place rectangles onto W x H sheets so that every per-sheet layout is
    // guillotine-legal and overlap-free, minimizing
    //   cost = (#sheets) * W * H - placed_area + 3 * unplaced_area.
    // Idea: each sheet is a pool of free rectangles; placement = best-fit free
    // rectangle + one guillotine split; search over the insertion order.

    // Print a feasible solution: m, then "idx sheet x y rot" per placed rect.
    printf("0\n");
    return 0;
}
```
