# Symmetric Emblem Compiler

## Problem
A plotter must ink a set of cells on an `N x N` grid (`N` even, cells indexed
`0..N-1` in each axis) so that the inked set is invariant under the 8 rigid
symmetries of the square about the grid's center (identity, the three
non-trivial rotations, and the four reflections). A list of **anchor cells**
that must end up inked is given; every cell that symmetry forces along with
an anchor must be inked too, but nothing else is required.

The 8 symmetries, indexed `0..7`, act on a cell `(x,y)` as:
```
0: (x,y)                       4: (N-1-x, y)      [mirror vertical axis]
1: (N-1-y, x)     [rotate 90]  5: (x, N-1-y)       [mirror horizontal axis]
2: (N-1-x, N-1-y) [rotate 180] 6: (y, x)            [main diagonal]
3: (y, N-1-x)     [rotate 270] 7: (N-1-y, N-1-x)    [anti-diagonal]
```
These formulas keep every result inside `[0,N)` for any even `N`, and they
compose to form the dihedral group of order 8 (`D4`).

You write a **straight-line plotter program** using 4 instruction types:
- `DOT x y` — ink cell `(x,y)` directly.
- `DEF k` ... `END` — define macro `k`. Macros are numbered `0,1,2,...` in
  `DEF` order (`k` must equal the count of macros already completed).
  Between `DEF k` and `END`, `DOT`/`CALL` instructions describe macro `k`'s
  *local shape* (cells relative to the grid frame). `DEF` blocks don't nest.
- `CALL k g` — apply symmetry `g` (`0<=g<=7`) to every cell of an
  **already-defined** macro `k` (`k` < the number of macros completed so
  far — only earlier macros, no self/forward reference, no cycles). At top
  level this inks the transformed cells; inside a `DEF` block it adds them
  to the macro being defined. A macro's local shape is fixed regardless of
  how later `CALL`s transform it.

**Cost.** `DEF`/`END` are free structural markers. Every `DOT` and every
`CALL` — top level or inside a macro body — costs exactly 1 instruction;
the objective is the total instruction count. Tokens may be split across
lines freely (the checker reads a flat whitespace-separated stream). At
most 1000 macros and 4000 total `DOT`/`CALL` instructions are allowed.

## Input (stdin)
```
N
A
x_1 y_1
...
x_A y_A
```
`N` even, `4 <= N <= 64`. `A` anchors follow, `0 <= x_i,y_i < N`
(`1 <= A <= 400`). Anchors may relate to each other through the required
symmetry — the input does not tell you which ones do.

## Output (stdout)
The plotter program as described above (a flat token stream).

## Feasibility
Let `S` be the final inked set (top-level `DOT`s plus top-level `CALL`
expansions). The output is feasible iff:
1. Every anchor is in `S`.
2. `S` is invariant: for every symmetry `g` in `0..7` and every cell `c` in
   `S`, applying `g` to `c` also lands in `S`.

Any parse error, out-of-range coordinate, bad macro/transform reference,
unterminated `DEF`, or instruction-budget overrun scores 0. Any violation of
(1) or (2) scores 0.

## Objective
Minimize the total instruction count.

## Scoring
The checker independently computes `B`, the size of the **orbit closure**
of the given anchors (apply all 8 symmetries to every anchor and take the
union) — the cell count a fully-unrolled, macro-free program would need.
With `R` your instruction count on a feasible output:
```
Ratio = min(1, 0.1 * B / R)
```
Fewer instructions is better; matching the unrolled baseline scores `0.1`.

## Constraints
- `4 <= N <= 64` (even), `1 <= A <= 400`, `0 <= x_i,y_i < N`.
- Deterministic scoring; no timing.

## Example
`N=8`, and the anchor list happens to contain the full 8-cell orbit of
`(0,1)` (namely `(0,1),(6,0),(7,6),(1,7),(7,1),(0,6),(1,0),(6,7)`), two
members of the 8-cell orbit of `(2,3)` (namely `(2,3),(4,2)`), and one
member of the 8-cell orbit of `(0,3)`, for `A=11` anchors total. The three
orbits are disjoint, so `B = 24`.

A program that just lists all 11 anchors as `DOT`s inside one macro and
calls it under all 8 symmetries costs `R=11+8=19`,
`Ratio=min(1,0.1*24/19)=0.126`. A program that notices the 11 anchors come
from only 3 underlying orbits, keeps one representative cell per orbit
(e.g. `(0,1)`, `(2,3)`, `(0,3)`) inside the macro, and calls it under all 8
symmetries costs `R=3+8=11`, `Ratio=min(1,0.1*24/11)=0.218` — fewer seed
cells buy a cheaper program without losing any required coverage.
