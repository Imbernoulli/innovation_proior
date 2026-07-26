# The Conservator's Loom: Rank-Faithful Tapestry Restoration

## Problem

A conservator is restoring an ancient m x m tapestry. Every cell of the weave
is an integer "thread code" in **Z_p** (arithmetic modulo a fixed prime `p`),
exact modular arithmetic, not real numbers -- two codes are either equal mod
`p` or they are not, no notion of "close". Age has torn some threads away
entirely: those cells are unknown and marked `?`. Other cells survived but
were stained by dye bleed at isolated points, so their recorded code may
disagree with the tapestry's underlying weave pattern -- you cannot tell a
stain from a clean cell just by looking at it.

Your job: output a **complete** m x m matrix over Z_p that (a) agrees with
every surviving (non-`?`) cell exactly, including stains, and (b) is as
*algebraically coherent* as possible -- has the **smallest possible rank
over the field Z_p** (exact modular Gaussian elimination). A genuine weave
uses only a handful of interacting thread patterns, so a faithful
restoration is low rank; a restoration stitched together cell by cell
without regard for the global pattern will generically come out full rank,
because rank is a *global* invariant no local, per-cell choice can see.

## Input (stdin)

```
m p
row_1
row_2
...
row_m
```
`m` is the tapestry side length, `p` is the (fixed) prime modulus. Each of the
next `m` lines has `m` space-separated tokens: either an integer in `[0,p-1]`
(a surviving cell) or the character `?` (a torn cell).

## Output (stdout)

Exactly `m` lines, each with `m` space-separated integers in `[0,p-1]`: your
complete restoration of the tapestry, row by row.

## Feasibility

Your output must have exactly `m*m` integer tokens, each a finite integer in
`[0,p-1]`. For every cell that was **not** `?` in the input, your output must
equal the given value exactly (including stained cells -- you must reproduce
them verbatim, you just should not let them leak into the cells you are
filling in). Any violation scores `0`.

## Objective (minimize)

`F` = the rank over Z_p of your complete output matrix, computed by exact
modular Gaussian elimination (no rounding, no tolerance).

## Scoring

The checker also computes `B`, the rank of its own naive baseline (every torn
cell filled with `0`). Your ratio is `ratio = min(1.0, B / F)`. Lower `F` (a
more coherent, lower-rank restoration) scores higher. Filling every torn
cell with `0`, or with the row average of the surviving cells, reproduces
(or barely beats) `B` -- a low score. A restoration that discovers the
tapestry's true few-pattern structure and propagates it algebraically to the
torn cells achieves a much smaller `F`, and a much higher score. Ten test
cases of increasing size are run; your final score is their mean.

## Constraints

`14 <= m <= 42`. `p` is a fixed prime a little above one million. Roughly
55% of the ordinary cells are given; a small block of rows always survives
completely intact (no `?`, never stained) -- a reliable witness to the
tapestry's true pattern, if you notice and use it. A handful (3-5) of cells
elsewhere carry a stain; these are always among the surviving cells, so you
must reproduce their exact value, but nothing requires you to reuse a
stained value anywhere else. The number of underlying thread patterns ("a
few") is **not** fixed across tapestries -- it varies instance to instance
and is never told to you, so assuming a single fixed count is a trap; the
intact rows let you test a candidate count directly (a count is right only
if it explains essentially all of a row's surviving cells at once, not just
a handful).

## Example (illustrative only, tiny 4x4, not to scale)

Suppose `m=4` and the true pattern is rank 1, `M[i][j] = a_i * b_j mod p`, no
stains. Correctly recovering the single vector pair `(a,b)` up to scale makes
your restoration rank 1, matching the true structure: with `B` (all-zero
baseline) at rank 3 or 4 typically, you would score `min(1, B/1) = 1.0`.
Filling torn cells with `0` instead keeps `F` near `B`, scoring close to the
`~0.1` baseline ratio used to calibrate this benchmark. This tiny example
only illustrates the mechanics; real instances (m >= 14) use a small, varying
number of thread patterns plus isolated stains, so the achievable minimum
`F` sits a few points above the true count, not necessarily 1.
