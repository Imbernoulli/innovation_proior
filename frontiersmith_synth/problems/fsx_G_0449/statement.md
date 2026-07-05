# Composing a Graphics Transform Chain with Fewest Scalar Multiplies

## Problem
A rendering pipeline composes a chain of linear transforms

    T = S_1 · S_2 · ... · S_L        (ordinary matrix product, applied in order)

Stage `S_i` maps an attribute space of width `d_i` to one of width `d_{i+1}`.
Because positions, normals, tangents, uv-coordinates and blend weights all have
different widths, the dimensions are **irregular**. Each stage is supplied in
one of three authentic forms:

- **DENSE** — an explicit `d_i × d_{i+1}` matrix `D`.
- **LOWRANK** — a rank-`r` deformation given only by its factors
  `U` (`d_i × r`) and `V` (`r × d_{i+1}`); the stage is `U·V` and is **never
  pre-multiplied for you**.
- **SUMLR** — a base transform plus a low-rank correction, `A + B·C`, with
  `A` (`d_i × d_{i+1}`), `B` (`d_i × r`), `C` (`r × d_{i+1}`).

Compute the single composite matrix `T`. You are free to reorder the product
(associativity), keep low-rank factors un-multiplied, and distribute sums
(distributivity) any way you like. Only the number of **scalar multiplications**
matters — matrix additions are free.

## Input (stdin)
```
L
d_0 d_1 ... d_L
```
then `L` stage blocks. Block `i` is one of:
```
DENSE
<d_i lines, each d_{i+1} ints>            # D
```
```
LOWRANK r
<d_i lines, each r ints>                  # U
<r lines, each d_{i+1} ints>              # V
```
```
SUMLR r
<d_i lines, each d_{i+1} ints>            # A
<d_i lines, each r ints>                  # B
<r lines, each d_{i+1} ints>              # C
```
All entries are integers. The **given matrices are numbered `0,1,2,...` in the
exact order they appear** (DENSE → `[D]`; LOWRANK → `[U,V]`; SUMLR → `[A,B,C]`).
These indices are the leaf ids of your program.

## Output (stdout) — a straight-line matrix program
```
K
<op> <a> <b>          (K lines)
```
- `op ∈ {MUL, ADD, SUB}`; `a`, `b` reference ids already defined (any given
  matrix `0..m-1`, or the result of an earlier line).
- Line `t` (0-based) produces a new value with id `m + t`.
- `MUL`: shapes `p×q` and `q×r` → `p×r`, costing `p·q·r` scalar multiplies.
- `ADD`/`SUB`: equal shapes → same shape, costing `0` scalar multiplies.
- The value produced by the **last** line must equal `T` exactly.

## Feasibility
The program must be well-formed (valid ids, shape-compatible ops, `1 ≤ K`) and
its final value must equal `T` entry-for-entry over exact integer arithmetic.
Any violation — bad schema, out-of-range id, shape mismatch, wrong result, or a
non-integer / `nan` / `inf` token — scores `0`.

## Objective
Minimize the total scalar-multiplication count `F` over all `MUL` lines.

## Scoring
Let `B` be the multiplies used by the naive construction (materialize each stage
densely in the given order, then fold left-to-right); the checker builds `B`
itself. With `F` your multiply count,

    Ratio = min(1, 0.1 · B / F).

Reproducing the naive construction gives `Ratio ≈ 0.1`; a 10× reduction caps at
`1.0`.

## Constraints
- `4 ≤ L ≤ 13`, dimensions up to `24`, `r ≥ 2`.
- `1 ≤ K ≤ 20000`; every intermediate matrix has `≤ 2^20` entries.
- Exact integer arithmetic throughout; nothing is ever timed.

## Example (worked score)
Suppose the naive construction uses `B = 12000` multiplies and your program
reorders the fold and absorbs a low-rank factor to reach `F = 4000`. Then
`Ratio = min(1, 0.1·12000/4000) = min(1, 0.3) = 0.3`. Pushing to `F = 1200`
(a 10× win) would cap the score at `1.0`.
