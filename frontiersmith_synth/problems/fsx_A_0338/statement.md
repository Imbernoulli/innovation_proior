# Quay Transfer Tensor: Minimal-Multiplication Crane Program

## Problem
A container terminal couples three resources: **vessels** (axis 0, `I` of them),
**yard blocks** (axis 1, `J` of them) and **time windows** (axis 2, `K` of them).
The planner's coupling model is an integer 3-D tensor `T` of shape `I x J x K`,
where `T[i][j][k]` is the net number of container moves that entangle vessel `i`,
yard block `j` and time window `k`.

The terminal executes the coupling with a **crane program**: a list of `R`
*separable stages*. Stage `r` is a triple of weight vectors
`(u_r in Q^I, v_r in Q^J, w_r in Q^K)` and contributes the rank-1 (separable)
tensor `u_r (x) v_r (x) w_r`, i.e. it adds `u_r[i] * v_r[j] * w_r[k]` to cell
`(i,j,k)`. Each stage costs exactly **one scalar multiplication** to evaluate the
associated bilinear map, so an `R`-stage program costs `R` multiplications.

Your job: reproduce `T` **exactly** with as few stages as possible.

## Input (stdin)
```
I J K
```
followed by the tensor entries, `I*J` lines, one line per `(i,j)` pair in row-major
order (`i` outer, `j` inner), each line holding the `K` integers
`T[i][j][0..K-1]`. All dimensions satisfy `2 <= I,J,K <= 5`.

## Output (stdout)
```
R
<stage 1>
...
<stage R>
```
Each `<stage>` is a single line of `I+J+K` rationals: the `I` entries of `u_r`,
then the `J` entries of `v_r`, then the `K` entries of `w_r`. Entries may be
integers or fractions `p/q` (exact rational arithmetic; no decimals, `nan` or
`inf`).

## Feasibility
The reconstructed tensor `sum_r u_r (x) v_r (x) w_r` must equal `T` in **every**
cell, evaluated exactly over the rationals. Any mismatch, malformed line, wrong
token count, or non-finite entry makes the whole submission infeasible.

## Objective
**Minimize** `R`, the number of stages (scalar multiplications).

## Scoring
Let `B` be the number of nonzero entries of `T` (the naive
one-multiplication-per-entry program). For a feasible submission with `R` stages,
```
Ratio = min(1, 0.1 * B / R)
```
Infeasible submissions score `0`. Reproducing the naive per-entry program scores
about `0.1`; a program ten times shorter caps the score at `1.0`.

## Constraints
- `2 <= I, J, K <= 5`.
- `1 <= R <= 20000`.
- All output entries are exact rationals.

## Example (worked score)
Suppose `T` is `2 x 2 x 2` with `B = 8` nonzero entries. A submission with
`R = 4` stages that reconstructs `T` exactly scores
`Ratio = min(1, 0.1 * 8 / 4) = 0.2`. A sharper `R = 2` program would score
`min(1, 0.1 * 8 / 2) = 0.4`.

## Notes on difficulty
The tensor is planted as a sum of separable stages whose count **exceeds every
axis length** (an overcomplete planted rank). Algebraic rank-recovery methods
(Jennrich / simultaneous diagonalization) require the rank to be at most a
dimension, so they cannot recover a minimal program. Slabbing along a single
axis and rank-factorizing each slab gives an easy upper bound; the best axis
sharpens it; but the true minimal `R` lies strictly below that bound and is not
known in closed form -- multiple search strategies (rational ALS, border-rank
constructions, cross-axis merging) can each push lower.
