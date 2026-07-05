# Glacier Sensor Net — Maximal D-Optimal Polarity Design

## Problem
A glacier monitoring campaign deploys an ice-penetrating radar array of `N`
sensor slots, each of which records one differential channel. Every channel is
wired with a fixed *polarity* — either `+1` or `-1`. Collecting the `N` readings
of `N` channels forms an `N x N` **polarity matrix** `M` whose entries are all
`+1` or `-1`.

For an experimental design like this, the *information volume* recovered from the
array is measured by the **absolute value of the determinant** `|det(M)|`
(the classical D-optimality criterion): the larger `|det(M)|`, the better the
array separates independent sub-surface signals.

Your task: **construct a `+/-1` polarity matrix `M` with the largest `|det(M)|`
you can find.**

`N` is always **odd**, so no Hadamard matrix of order `N` exists and there is no
known closed-form optimal construction — the maximal-determinant problem for odd
orders is an open computational search problem. The Hadamard bound
`N^(N/2)` is only an (unreachable) upper reference; it is never attained.

## Input (stdin)
A single line containing the odd integer `N`.

## Output (stdout)
`N` lines, each with `N` space-separated integers, every one equal to `+1`
(written `1`) or `-1`. This is the matrix `M`, row by row.

## Feasibility
The output is feasible iff it contains exactly `N*N` integer tokens, arranged as
`N` rows of `N` values, and **every entry is either `1` or `-1`**. Any other
token (`0`, `2`, a float, `nan`, `inf`, missing/extra entries) makes the output
infeasible and scores `0`.

## Objective
Maximize `|det(M)|`, the exact integer determinant computed by **Bareiss
fraction-free elimination** (no floating point in the determinant — the value is
an exact integer).

## Scoring
The checker computes the exact `|det(M)|` and an internal reference determinant
`D0 = |det(M0)|`, where `M0` is the trivial feasible design
`M0[i][j] = +1 if i==j else -1` (diagonal `+1`, off-diagonal `-1`). Scores are
compared in a size-normalized geometric scale to keep every order comparable:

```
F  = |det(M)| ** (2.0 / N)
B  = D0       ** (2.0 / N)
sc = min(1000, 100 * F / max(1e-9, B))
Ratio = sc / 1000
```

A design equal to the trivial reference scores `Ratio = 0.1`; a singular
(`det = 0`) or infeasible matrix scores `0`. There is no known construction that
saturates the scale, so headroom always remains.

## Constraints
- `15 <= N <= 33`, `N` odd.
- Entries strictly in `{-1, +1}`.
- Determinant computed exactly (Bareiss); scoring is bit-for-bit deterministic.

## Example
For `N = 3` a feasible matrix is
```
 1  1  1
 1 -1  1
 1  1 -1
```
with `det = 4`, `|det| = 4`. The trivial reference `M0` for `N = 3` is
`[[1,-1,-1],[-1,1,-1],[-1,-1,1]]` with `det = -4`, so `D0 = 4` and this matrix
would score `Ratio = 0.1`. (Illustrative sizes only; graded instances use
`15 <= N <= 33`.)
