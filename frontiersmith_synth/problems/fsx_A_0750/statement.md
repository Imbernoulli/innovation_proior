# Chambered Pocket Clearing: Fewest Counted Stamps

## Problem
A rectilinear pocket is marked on a unit grid. You clear it with a **stamp tool**: a tool
of size `s` places an `s x s` square block, and the block counts as covered only if its
top-left corner is chosen so the entire square lies inside the pocket. You are given a
fixed catalog of tool sizes (always includes size `1`, which fits anywhere the pocket
does). A **program** is an ordered list of stamps `(s, r, c)` — tool size and top-left
row/col. The pocket is cleared once every pocket cell is covered by at least one stamp
(stamps may overlap; only the union matters).

Mounting a different-size tool costs a fixed changeover fee `C`, charged every time
**consecutive stamps in your program's listed order** use different sizes. Using the same
tool for a long uninterrupted run is free after the first mount. Some parts of the pocket
are wide, admitting large tools; others are only one cell wide (barrier columns, comb
teeth) and can *only* ever be stamped with size `1`, no matter how the rest is cleared —
these are unavoidable finishing passes near tight, corner-like geometry. Your job is to
choose placements **and** their program order to minimize total cost.

## Input (stdin)
```
H W K C
s_1 s_2 ... s_K
```
then `H` lines of `W` characters: `#` marks a pocket cell, `.` marks a non-pocket cell.
`s_1 < s_2 < ... < s_K` are the available tool sizes, `s_1 = 1` always.

## Output (stdout)
```
N
s_1 r_1 c_1
...
s_N r_N c_N
```
`N` stamps, each an integer tool size and 0-indexed top-left row/col.

## Feasibility
Every stamp's size must be one of `s_1..s_K`. Every stamp's `s x s` square must lie fully
in bounds (`r+s <= H`, `c+s <= W`) and every cell in that square must be a pocket cell (`#`).
The union of all stamps must cover **every** pocket cell. Any violation, any malformed or
non-integer token, or empty output scores `0`.

## Objective
Let `N` be the stamp count and let `changes` be the number of indices `i` (2..N) where
`s_i != s_{i-1}` (a tool change, counted in your program's listed order — reordering stamps
that don't need to run in a particular pass is free strategy, not a fixed layout). Minimize

```
F = N + C * changes
```

## Scoring
Let `B` be the number of pocket cells (the cost of the naive program that stamps every
pocket cell individually with size `1`, in a single block — always feasible, `changes = 0`).
With your `F`:
```
Ratio = min(1, 0.1 * B / F)
```
The naive per-cell program scores `0.1`. Halving your cost roughly doubles the ratio. The
minimum achievable `F` is not known to be reachable by any polynomial construction, so
headroom remains above what careful covering achieves.

## Constraints
- `1 <= H, W <= 70`, `2 <= K <= 5`, `1 <= C <= 6`.
- `0 <= N <= 200000`.
- Deterministic scoring; checker runs in `O(H*W)`. Time limit 5s.

## Example
Suppose the pocket is a `3 x 7` strip with a single checkerboard barrier column at `c=3`
(so column 3 alternates `#`/`.` by row) and catalog `{1, 3}`. Any square of size `3`
straddling column 3 always hits a `.` cell there, so column 3 must be cleared with `1x1`
stamps regardless of strategy; each side (`3 x 3` each) can be cleared with a single size-3
stamp. A program that places both size-3 stamps first, then the barrier's size-1 stamps,
pays only **one** tool change; interleaving them (e.g. clearing left-to-right column by
column) would pay a change at every crossing. With `B` = 21 pocket-ish cells minus the
barrier's missing cells, the batched program scores substantially higher than an
unbatched one with the same stamp count — order alone moves the score.
