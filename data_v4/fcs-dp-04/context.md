# Counting glass-panel pavings of a pillared conservatory floor

## Research question

A botanical conservatory has a rectangular floor laid out as an `h x w` grid of unit
cells. Some cells hold structural pillars and can never be paved; every other cell is open
floor that **must** be covered. The gardeners pave the open floor with `1 x 2` glass panels,
each panel laid flat over two orthogonally adjacent open cells (horizontally or vertically).
Panels may not overlap, may not cover a pillar, and may not hang past the conservatory wall;
every open cell must be covered by exactly one panel.

Count the number of distinct ways to pave the entire open floor, reported modulo a given
integer `p`. Two pavings differ if some open cell is covered by panels oriented or positioned
differently. If the open floor cannot be paved at all (for instance, it has an odd number of
open cells), the count is `0`.

This is the planar **domino-tiling enumeration** problem on a grid with holes. The grid is
short but very wide (`h <= 14`, `w <= 1000`), which is exactly the regime where the contour
between paved and unpaved territory — not the full set of placed panels — is the only state
that matters.

## Input / output contract

- Input (stdin):
  - The first line holds three integers `h w p`
    (`1 <= h <= 14`, `1 <= w <= 1000`, `1 <= p <= 10^9`).
  - The next `h` lines each hold a string of length `w` over the alphabet `{'.', '#'}`.
    Row `r` (0-indexed from the top), column `c` (0-indexed from the left): `'.'` is open
    floor that must be covered, `'#'` is a pillar that must stay uncovered.
- Output (stdout): a single line with the number of complete pavings of the open floor,
  taken modulo `p`. When `p = 1` the answer is `0` (everything is congruent to `0`).
- Time limit: 2 seconds. Memory: 256 MB.

Example: for the `3 x 4` floor
```
#...
....
...#
```
(pillars at the top-left and bottom-right corners, ten open cells) there are exactly `5`
complete pavings, so the output is `5`.

## Background

The brute-force picture is direct: walk the open cells in some fixed order, and for the first
still-uncovered open cell try to extend it with a horizontal or vertical panel, recursing.
That backtracking is obviously correct and is the reference oracle, but it explores a search
tree whose size is exponential in the number of open cells; on a full `h x w` board it is
hopeless once `h * w` grows past a couple dozen.

A second, equally tempting line of attack is a column-by-column dynamic program whose state is
"which cells of the current column boundary stick out into the next column." The open question
that decides the whole design is how to encode that boundary compactly and update it without
ever materializing a placement of every panel:

- **Whole-column profile DP.** Move one full column at a time; the state enumerates every way
  the previous column's panels protrude into the current one. Transitioning between two columns
  means enumerating compatible pairs of `h`-bit masks, which costs up to `3^h`-ish work per
  column boundary and is fiddly to get right for vertical panels inside a column.
- **Cell-by-cell contour DP.** Move one *cell* at a time along a fixed sweep order and keep the
  jagged "broken" boundary as a single bitmask, deciding only the panel that newly covers the
  current cell. The open question is what each bit means as the sweep crosses a cell, and how a
  horizontal versus a vertical panel rewrites exactly one bit of the mask.

## Evaluation settings

Judged on hidden tests covering: tiny boards checked against backtracking; boards with an odd
number of open cells (answer `0`); fully blocked boards (`#` everywhere, the empty paving, count
`1`); boards split into independent regions by walls of pillars; thin `1 x w` and `2 x w`
strips with known closed forms (a `2 x w` open strip has `Fib(w+1)` pavings); the largest boards
`h = 14`, `w = 1000` for both speed and `64`-bit-safe modular accumulation; and assorted moduli
including `p = 2`, small primes, and large primes such as `998244353` and `10^9 + 7`, plus the
degenerate `p = 1`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int h, w;
    long long p;
    if (!(cin >> h >> w >> p)) return 0;

    vector<string> grid(h);
    for (int r = 0; r < h; r++) cin >> grid[r];

    // TODO: count complete 1x2 pavings of the '.' cells (modulo p), where '#'
    //       cells must stay uncovered and every '.' cell must be covered.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
