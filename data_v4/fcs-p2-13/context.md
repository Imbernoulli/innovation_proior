# Largest all-ones square submatrix in a binary grid

## Research question

You are given a binary grid with `H` rows and `W` columns; every cell is either `0` or `1`. Among all
axis-aligned **square** submatrices whose cells are *all* `1`, find the one with the largest area and
output that area (side length squared). If the grid contains no `1` at all, the answer is `0`.

A "square submatrix" is a contiguous block of `k` consecutive rows and `k` consecutive columns for some
`k >= 1`; it is valid only if every one of its `k*k` cells equals `1`. We want the maximum `k*k` over all
valid squares.

The grid can be as large as `1500 x 1500`, so the algorithm needs to run in close to linear time in the
number of cells.

## Input / output contract

- Input (stdin): the first two tokens are `H` and `W` (`1 <= H, W <= 1500`). Then follow `H*W` integers,
  each `0` or `1`, given in row-major order (row 0 first, then row 1, ...). Tokens are whitespace-separated;
  the exact spacing and line breaks do not matter to the parser.
- Output (stdout): a single line with one integer — the area (`side * side`) of the largest all-ones
  square submatrix, or `0` if there is none.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for the `3 x 4` grid

```
1 1 0 1
1 1 1 1
0 1 1 1
```

the answer is `4`: the largest all-ones square is `2 x 2` (e.g. rows 1-2, columns 2-3), area `4`. There is
no all-ones `3 x 3` square anywhere, and although there are wide runs of `1`s, area counts only square
blocks.

## Background

Two families of approach are on the table before committing to one:

- **Greedy / area-scan heuristic.** Find the densest or largest all-ones *region* — for instance, locate
  the widest all-ones rectangle (via the classic histogram-stack method) or the cell that "looks"
  surrounded by the most `1`s, then read a square out of it. These ideas are appealing because the
  all-ones-rectangle machinery is well known and fast, and intuitively the biggest square should live
  inside the biggest dense block. The open question is whether "biggest rectangle" or any local
  density signal actually pins down the biggest *square*, or whether it can systematically mislead.
- **Dynamic programming.** Some formulation of dynamic programming, tracking a local notion of square
  size as the grid is scanned, may also apply. What exactly to track at each cell, what recurrence (if
  any) relates it to nearby cells, and whether it is enough to pin down the true largest square, are all
  open.

## Evaluation settings

Judged on hidden tests covering: all-zeros grids, all-ones grids, single-row and single-column grids,
grids where the largest all-ones rectangle differs structurally from the largest all-ones square,
squares planted inside random noise, and full-size `1500 x 1500` random grids.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int H, W;
    if (!(cin >> H >> W)) return 0;

    // The grid has H*W cells, each 0 or 1, in row-major order on stdin.

    // TODO: compute the area (side * side) of the largest all-ones square submatrix.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
