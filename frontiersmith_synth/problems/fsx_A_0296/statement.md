# Subway Signal Polarity: Maximum-Determinant Wiring

## Problem

A metro operator runs a subway network of `N` stations. Every ordered pair of
stations `(i, j)` carries a signalling line whose **polarity** is either `+1` or
`-1`. Collect the polarities into an `N x N` matrix `A` with `A[i][j] in {-1,+1}`.

The network's *decoding robustness* is measured by the **absolute determinant**
`|det(A)|`: a larger determinant means the station-to-station signals are more
linearly independent and the control system is harder to jam. Your job is to
choose the polarities to make `|det(A)|` as large as possible.

Some lines are pinned by existing hardware and cannot be changed:

- Every station's self-line is fixed: `A[i][i] = +1`.
- A list of additional off-diagonal cells is fixed to a given polarity.

All remaining cells are free for you to choose.

The station counts are **odd** (`13, 15, ... , 27`), so no Hadamard matrix of that
order exists; the true maximum determinant for these orders is not known in closed
form, and many different wirings come close — this is an open-ended construction
problem, not a puzzle with a memorizable answer.

## Input (stdin)

```
N seed K
i1 j1 v1
i2 j2 v2
...
iK jK vK
```

- `N` — number of stations (matrix side).
- `seed` — an integer used only by the reference/default wiring; you may ignore it.
- `K` — number of fixed cells that follow.
- Each of the next `K` lines gives a fixed cell: 0-indexed `i j`, and its
  required polarity `v` in `{-1, +1}`. The `N` diagonal cells (`v = +1`) are
  always among these.

## Output (stdout)

Print the `N x N` matrix, one row per line, `N` space-separated values per row,
each value being `-1` or `+1`. Every fixed cell listed in the input **must** hold
its required value.

## Feasibility

Your output is rejected (score `0`) unless it contains exactly `N*N` integer
tokens, every token is `-1` or `+1`, and every fixed cell equals its required
polarity. A singular matrix (`det = 0`) also scores `0`.

## Objective

Maximize `|det(A)|`, computed **exactly** by integer Bareiss elimination (no
floating-point rounding in the determinant).

## Scoring

The checker rebuilds an internal **baseline** wiring `A0` (the fixed cells plus a
default pseudo-random fill of the free cells) and scores the *log-determinant
excess* of your matrix over that baseline:

```
q(M)  = log2(|det M|)
L0    = q(A0) - DELTA          # DELTA = 4.0, fixed
F     = q(A) - L0
B     = DELTA
Ratio = min(1000, 100 * F / B) / 1000    # clamped to [0, 1]
```

- Reproducing the baseline determinant scores exactly `Ratio = 0.1`.
- Every doubling of `|det|` above the baseline adds `1 / DELTA` of the way up;
  because the score grows with the *logarithm* of the determinant, it stays
  graded and never trivially saturates.
- The score is fully deterministic: same matrix, same score.

## Constraints

- `13 <= N <= 27`, `N` odd.
- All arithmetic on your side may be done in exact integers.
- Time limit `5 s`, memory `512 MB`.

## Example (worked score)

Suppose for some instance the baseline determinant is `|det A0| = 2^30` and you
submit a matrix with `|det A| = 2^42`. Then `q(A) - q(A0) = 12` bits, so
`F = 12 + DELTA = 16`, `B = DELTA = 4`, `Ratio = min(1000, 100*16/4)/1000 =
min(1000, 400)/1000 = 0.4`. Merely matching the baseline (`|det A| = 2^30`) would
give `F = B = 4` and `Ratio = 0.1`.
