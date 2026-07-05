# Near-Orthogonal Latin Squares: Maximizing Satisfied Orthogonality

## Problem

In statistical **experiment design**, a set of mutually orthogonal Latin squares (MOLS)
encodes a factorial layout in which every combination of two treatment factors occurs
exactly once. For many orders a *perfect* orthogonal set does not exist or is unknown
(famously there is **no** pair of orthogonal Latin squares of order 6 — Euler's 36
officers — and whether **three** MOLS of order 10 exist is *open*). When a perfect design
is unreachable, the practitioner wants the design that is *as close to orthogonal as
possible*: one that realizes as many distinct factor-combinations as possible.

You are given an order `n` and a count `k`. Construct `k` Latin squares of order `n`
so that, when any two of them are superimposed, the number of **distinct** ordered
symbol-pairs is as large as possible — summed over every pair of squares.

A **Latin square of order `n`** is an `n x n` array with entries in `{0, 1, ..., n-1}`
in which every row and every column is a permutation of `{0, ..., n-1}`.

## Input (stdin)

One line:

```
n k
```

- `n` — the order of each Latin square (`6 <= n <= 22`).
- `k` — the number of Latin squares to construct (`2 <= k <= 3`).

## Output (stdout)

Print `k` Latin squares as `k * n` lines of `n` space-separated integers each. The output
is read as `k` consecutive blocks of `n` rows; block `m` (0-indexed) is the `m`-th square.
Blank lines and extra whitespace are ignored; the checker reads exactly `k*n*n` integer
tokens in row-major, square-by-square order.

## Feasibility

The output is rejected (score `0`) unless:

- there are exactly `k*n*n` integer tokens, each a finite integer in `[0, n-1]`;
- each of the `k` squares is a valid Latin square (every row and every column is a
  permutation of `{0, ..., n-1}`).

Any `nan`/`inf`/non-integer token, wrong token count, or non-Latin square scores `0`.

## Objective (maximize)

For two Latin squares `A`, `B`, let

```
D(A, B) = | { (A[i][j], B[i][j]) : 0 <= i,j < n } |
```

be the number of **distinct** ordered pairs seen when they are superimposed. Then

```
F = sum over all unordered pairs p < q of  D(L_p, L_q).
```

`F` is maximized (`= C(k,2) * n^2`) exactly when the set is mutually orthogonal — which is
**unreachable** for every `(n, k)` in this task, so `F` measures how near-orthogonal your
design is.

## Scoring

The checker builds an internal baseline `B` = the value of `F` for the standard **cyclic**
construction `L_m[i][j] = (a_m * i + j + s_m) mod n` (with `a_m` coprime to `n`). For the
orders used here every residue coprime to `n` is even, so no cyclic pair is orthogonal —
the baseline is a genuine, beatable starting point. Your score is

```
Ratio = min(1000, 100 * F / B) / 1000
```

so reproducing the cyclic baseline yields `~0.1`, and a set with `10x` the baseline's
distinct-pair total would cap at `1.0`. There is no known optimal construction for these
orders, so the ceiling is genuinely open.

## Example

Let `n = 3`, `k = 2` (illustrative only — small prime orders are *excluded* from the graded
ladder because their cyclic squares are already perfectly orthogonal). Output

```
0 1 2
1 2 0
2 0 1

0 1 2
2 0 1
1 2 0
```

Superimposing the two squares gives the 9 pairs
`(0,0)(1,1)(2,2) / (1,2)(2,0)(0,1) / (2,1)(0,2)(1,0)` — all 9 distinct, so `D = 9 = n^2`
and `F = 9`. This pair is orthogonal; for `n in {6,10,12,14,18,22}` such perfection is
impossible or unknown, and maximizing `F` is the open problem.

## Constraints

- `6 <= n <= 22`, `2 <= k <= 3`.
- Scoring is exact integer arithmetic and fully deterministic.
