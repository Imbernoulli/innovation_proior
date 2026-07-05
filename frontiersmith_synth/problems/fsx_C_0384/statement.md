# Freight-Yard Block Assignment: Maximum Restricted-Latin Completion

## Problem
A railway classification yard is organised as an `n x n` board: `n` classification
**tracks** (rows) and `n` shunting **shifts** (columns). There are `n` outbound
**destination blocks** `0, 1, ..., n-1`. A cut of cars for exactly one destination block
may be built at each `(track, shift)` cell.

Two operational rules must never be violated:

> **Conflict rule (Latin).** A destination block may be built at most once per track
> (row) and at most once per shift (column) -- otherwise two identical cuts collide on
> the same lead or ladder.

So the built cells form a (partial) **Latin square** over the `n` blocks.

In addition, each cell carries physical restrictions:

> **Gauge rule (forbidden list).** Cell `(i, j)` has a set of **forbidden blocks**
> `Fb[i][j]` (loading-gauge / coupler / weight-limit incompatibilities). A cut for a
> forbidden block may not be built at that cell.

Some cells are already **pre-built** givens (existing consists); the givens satisfy both
rules and none uses a forbidden block. You must build cuts at as many additional cells as
possible **without violating either rule** and **without altering any given**. Cells may be
left empty.

Deciding whether a partial Latin square can be completed is NP-complete; adding per-cell
forbidden lists (a list-restricted / "list-colouring" Latin completion) only makes it
harder. There is no simple closed-form optimum: greedy fills dead-end, while
constraint-ordered search installs strictly more.

## Input (stdin)
```
n
grid row 0 : n tokens        <- pre-built givens
...
grid row n-1 : n tokens
forbid row 0 : n tokens      <- forbidden lists
...
forbid row n-1 : n tokens
```
- A **grid** token is either a block index `0..n-1` (a pre-built cut) or `.` (empty).
- A **forbid** token is either `-` (nothing forbidden at this cell) or a comma-separated
  list of distinct block indices, e.g. `0,3,5` (those blocks are forbidden here).
- The givens form a valid restricted partial Latin square (no given conflicts; no given
  uses a forbidden block).

## Output (stdout)
Print the completed board: `n` lines, each with `n` whitespace-separated tokens. Each token
is either a block index `0..n-1` (a built cut) or `.` / `-1` (cell left empty).

## Feasibility
Your board is rejected (score `0`) if any of these hold:
- the output does not contain exactly `n*n` tokens, or a token is not `.`/`-1`/an integer;
- any built value is not an integer in `[0, n-1]`;
- any given cell is altered or emptied;
- any block repeats within a row or within a column (among built cells);
- any built cell uses a block that is forbidden at that cell.

## Objective (maximize)
`F` = the number of built (non-empty) cells in your feasible board (givens included).

## Scoring
Let `B` = the number of pre-built givens (the "build nothing new" baseline). The score is
```
sc    = min(1000, 100 * F / B)
Ratio = sc / 1000
```
so echoing only the givens scores `0.1`. Scoring is exact integer arithmetic and fully
deterministic.

## Constraints
- `4 <= n <= 9` (small scale).
- Given density roughly `0.42 - 0.50` of the cells.
- Up to `floor(n/2)` blocks forbidden per cell.

## Example
Input:
```
3
0 . 2
. . .
2 . 0
- 1 -
0 - -
- - -
```
Here `B = 4` givens, so echoing the input unchanged gives `F = 4`, `Ratio = 0.1`.
Cell `(0,1)` forbids block `1`; cell `(1,0)` forbids block `0`. A valid restricted
completion (e.g. filling `(0,1)=1` is forbidden, so `(0,1)` must be built as... none of the
remaining legal blocks fit, leaving it empty) shows why forbidden lists block naive fills.
Building the legal cells `(1,1)=... ` up to `F = 7` would give
`sc = min(1000, 100*7/4) = 175`, `Ratio = 0.175`. (Illustrative sizes; the real ladder uses
`n >= 4`.)
