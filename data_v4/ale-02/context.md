# Grid Polyomino Packing

## Research question

A rectangular floor of `H x W` unit cells must be tiled with stock parts. You are
given a small catalogue of part *shapes* — each shape is a polyomino (a connected
set of unit cells), supplied with a bound on how many copies of that shape you may
use. Every placed part may be rotated by 0, 90, 180 or 270 degrees and dropped
anywhere on the floor, as long as it stays fully inside the floor and does not
overlap any previously placed part. Unused floor cells are wasted.

Maximise the number of **covered cells** — equivalently, since parts never
overlap, the total area of the parts you place. The shapes rarely tile the floor
exactly, so there is no closed-form optimum: this is a maximum-coverage packing
problem, NP-hard in general, judged by a continuous coverage score. Choosing
*which* parts to place, *where*, and in *which rotation* — and being willing to
pull a badly-placed part back out to make room for a better arrangement — is the
whole game.

## Input / output contract

Input (stdin), all whitespace-separated integers:

- A line `H W K`: floor height `H`, width `W`, and the number of distinct shapes
  `K`.
- Then `K` shape blocks. Block `k` (0-indexed) begins with `A_k cnt_k`: the number
  of cells `A_k` of shape `k` and the maximum number of copies `cnt_k` you may
  place. The next `A_k` lines each give a cell `r c` as a non-negative offset; the
  block lists the base orientation, normalised so its minimum row and minimum
  column are both 0.

Output (stdout):

- A line `P`: the number of placements (`P >= 0`).
- Then `P` lines `k rot ar ac`: place one copy of shape `k`, rotated `rot * 90`
  degrees clockwise (`rot` in `0..3`), with the rotated-and-renormalised shape's
  top-left corner at floor cell `(ar, ac)` (row, column, 0-indexed).

Rotation convention: to rotate a cell `(r, c)` by 90 degrees clockwise, map it to
`(c, -r)`; after applying `rot` such steps, re-normalise the whole shape so its
minimum row and column are 0, then anchor it at `(ar, ac)`.

Constraints (instances): `12 <= H, W <= 30`; `4 <= K <= 8`; each `A_k` in
`1..6`; `cnt_k` generous (typically enough copies to over-fill the floor on their
own, so the floor — not the inventory — is the binding constraint). Time limit:
about 2 seconds. Memory: 256 MB.

## Background

Two ideas frame the approach before committing to one.

- **First-fit greedy.** Sort shapes largest-area-first; for each shape and each
  rotation, sweep all anchor positions and drop a copy wherever it currently fits.
  This is fast and always feasible, but it is a one-shot construction: once a cell
  is taken it is never reconsidered, so a few early, poorly-aligned big parts can
  strand pockets of empty cells that nothing left in the catalogue can fill.

- **Local search over the placement multiset.** Keep a set of current placements
  and repeatedly perturb it — add a part, remove a part, or swap one for another —
  accepting some coverage-reducing moves so the search can vacate a wasteful part
  and re-tile the freed region. This can climb out of the dead ends greedy walks
  into, but only if the per-move cost is tiny, because thousands of moves per
  millisecond are needed for it to pay off.

The open question is how to make each perturbation — in particular the collision
test against everything already on the floor — cheap enough that a metaheuristic
is worthwhile.

## Evaluation settings

**Score.** The local scorer (`verify/score.py`) reads the instance and the
solution, replays every placement, and reports the number of distinct covered
cells. Because feasible placements never overlap, this equals the summed area of
all placed parts.

**Feasibility -> 0 floor.** The score is `0` (the worst possible) if the output is
infeasible for any reason: it is malformed or truncated; a placement names a
non-existent shape or a rotation outside `0..3`; a placement pokes outside the
`H x W` floor; two placed cells coincide (overlap); or more than `cnt_k` copies of
some shape `k` are used. An empty solution (`P = 0`) is feasible but covers `0`
cells, so it ties the infeasible floor. The reference baseline the coverage ratio
is reported against is the first-fit greedy above.

**Instances.** `verify/gen.py SEED` draws `H, W`, a library of `K` random
connected polyominoes (grown by random accretion, areas `1..6`), and a generous
copy bound per shape, then prints the instance. Seeds are independent, so a fixed
seed set (e.g. `1..20`) gives a stable benchmark.

## Code framework

A single self-contained C++17 program reading stdin and writing stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int H, W, K;
    if (scanf("%d %d %d", &H, &W, &K) != 3) return 0;

    // Read K shape blocks: "A_k cnt_k" then A_k lines "r c".
    vector<vector<pair<int,int>>> base(K);
    vector<int> cnt(K);
    for (int k = 0; k < K; k++) {
        int A; scanf("%d %d", &A, &cnt[k]);
        base[k].resize(A);
        for (auto &cell : base[k]) scanf("%d %d", &cell.first, &cell.second);
    }

    // TODO: choose placements (shape, rotation, anchor) that stay in-grid,
    //       respect the counts, never overlap, and maximise covered cells.
    //       Always print a FEASIBLE solution (P = 0 is feasible).

    vector<array<int,4>> placements;   // each: {k, rot, ar, ac}
    printf("%d\n", (int)placements.size());
    for (auto &p : placements)
        printf("%d %d %d %d\n", p[0], p[1], p[2], p[3]);
    return 0;
}
```
