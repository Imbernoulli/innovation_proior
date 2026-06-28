# Polyomino Tiling Coverage

## Research question

You are given an `H x W` board, every cell initially empty, and `P` polyomino piece
**types**. Type `t` is a small connected shape described by `k_t` integer cell
**offsets** (already normalized so the shape touches the top and left of its
bounding box), and may be used at most `avail_t` times. A **placement** picks a
type, one of its 4 rotations (quarter-turns clockwise), and an anchor cell; it
occupies the rotated, re-normalized cells translated to the anchor. Placements must
stay on the board and never overlap. The task is to choose a set of placements that
**maximizes the number of covered (occupied) board cells**. This is a packing /
exact-cover-style problem: it is NP-hard, there is no optimum to read off, and the
quality of a solution is a continuous coverage count, so the only lever is the
heuristic that decides which pieces to place where.

## Input / output contract

- Input (stdin):
  - first line: `H W P` — board height `H` and width `W` (`12 <= H, W <= 26`, and
    `W <= 60` so a board row fits in one 64-bit word), and the number of piece
    types `P` (`4 <= P <= 9`);
  - then, for each of the `P` types in turn:
    - a line `k` — the number of cells (`3 <= k <= 6`);
    - `k` lines, each `dr dc` — an integer cell offset of the shape; the offsets are
      normalized so `min(dr) = 0` and `min(dc) = 0`;
    - a line `avail` — how many copies of this type may be used (`avail >= 0`).
- Output (stdout):
  - first line: `M` — the number of placements (`M >= 0`);
  - then `M` lines, each `type rot r c` — place piece `type` (`0 <= type < P`) with
    rotation `rot` (`rot in {0,1,2,3}`, quarter-turns clockwise) anchored at board
    cell `(r, c)`.
- Time limit: about 2 seconds. Memory: 256 MB.

The occupied cells of a placement `(type, rot, r, c)` are obtained by rotating
type `type`'s offsets by `rot` quarter-turns clockwise (the map `(r, c) -> (c, -r)`
applied `rot` times), re-normalizing the rotated offsets so their minimum row and
column are both `0`, and then translating by `(r, c)`: if the rotated+normalized
offsets are `{(dr, dc)}`, the occupied cells are `{(r + dr, c + dc)}`.

## Background

This is the discrete cousin of strip packing and bin packing: we are tiling a fixed
rectangular region with an inventory of small polyominoes and we are paid for the
area we manage to cover. Two textbook reference points frame the design.

- **Largest-piece-first greedy.** Sort the types by decreasing cell count and, for
  each type while copies remain, scan the board top-to-bottom / left-to-right and
  drop the piece (in whichever of its 4 rotations fits) at the first empty anchor.
  This is fast and never overlaps, but committing big pieces early carves the empty
  region into awkward pockets that the remaining inventory cannot fill, so it leaves
  a ragged uncovered fringe. It is the natural normalizer the scorer measures
  against.
- **Row-bitmask board + local search.** Because `W` is small, a whole board row is a
  single machine word and a piece's footprint is a handful of row-masks. Testing
  "does this placement fit here?" becomes a few word-`AND`s, and adding/removing a
  placement is a few word-`OR`/`AND`-`NOT`s. That makes each candidate move cheap
  enough to run a real metaheuristic — add/remove simulated annealing with
  ruin-and-recreate — that repairs the greedy's holes by tearing out a few pieces
  and refilling, exploring far more configurations per second than any per-cell
  bookkeeping would allow. The board-as-bitmap representation is what turns an
  expensive overlap-and-coverage recomputation into an `O(rows-of-piece)` step.

## Evaluation settings

A solution is first checked for **feasibility**; any violation floors the score to
**0**:

1. the output parses as a leading integer `M >= 0` followed by exactly `4*M`
   further integers (`M` lines of `type rot r c`);
2. every `type` is in `[0, P)` and every `rot` is in `{0, 1, 2, 3}`;
3. every occupied cell of every placement is on the board:
   `0 <= row < H` and `0 <= col < W`;
4. no two occupied cells, across all placements, coincide (no overlap; the pieces
   tile, they do not stack);
5. each type `t` is used at most `avail_t` times.

For a feasible solution the **coverage** is the number of distinct occupied board
cells (equivalently the sum of the placed pieces' sizes, since there is no overlap).
Higher is better. The score normalizes against the deterministic **greedy
largest-piece-first** baseline that the scorer recomputes itself:

```
score = round(1_000_000 * solver_coverage / max(1, baseline_coverage))     (0 if INFEASIBLE)
```

So the greedy baseline scores about `1_000_000`; covering more cells scores more, an
infeasible output scores `0`.

**Instances** are generated deterministically from an integer seed. `H` and `W` are
in the low tens; `P` types are drawn, each a small polyomino of 3–6 cells grown by a
random walk and normalized, with a handful of copies available. The available area
(`sum_t avail_t * k_t`) is bumped to comfortably exceed `H*W`, so full coverage is
in principle possible — the difficulty is the packing, not a shortage of pieces.
This is exactly the regime where a strong add/remove local search beats a greedy
that paints itself into uncoverable corners.

## Code framework

A single self-contained C++17 program that reads the instance from stdin and writes
a feasible solution to stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int H, W, P;
    if (!(cin >> H >> W >> P)) return 0;
    vector<vector<pair<int,int>>> shape(P);   // normalized offsets per type
    vector<int> avail(P);
    for (int t = 0; t < P; ++t) {
        int k; cin >> k;
        shape[t].resize(k);
        for (int i = 0; i < k; ++i) cin >> shape[t][i].first >> shape[t][i].second;
        cin >> avail[t];
    }

    // A feasible fallback: output nothing (M = 0). It is always valid (no overlap,
    // nothing out of bounds) but covers nothing.
    // TODO heuristic: keep the board as H 64-bit row-words; precompute each type's
    // 4 distinct rotations as row-masks; greedily fill (largest piece first) for a
    // feasible start, then run add/remove simulated annealing with a popcount-delta
    // coverage update and periodic ruin-and-recreate kicks. Every state is
    // overlap-free and in-bounds by construction, so any snapshot is feasible.
    cout << 0 << '\n';
    return 0;
}
```
