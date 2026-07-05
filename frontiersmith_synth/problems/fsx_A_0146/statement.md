# Excavation Grid Resonance

## Problem

An archaeological site is laid out as an `N x N` **dig grid**. Each cell holds a
buried artifact whose magnetic orientation is either **+1** or **-1**. Some cells
have already been excavated, so their orientation is *known and fixed*. The
remaining cells are still buried and their orientation is yours to assign.

The survey team measures the site's **structural resonance** as the absolute
value of the determinant of the `N x N` orientation matrix. A larger
`|det|` means a more informative excavation plan. Choose the orientations of the
still-buried cells to make the resonance as large as possible.

`N` is **odd** for every instance, so no perfectly balanced (Hadamard) layout
exists and the maximum achievable resonance is not known in closed form — you
must search for a strong layout.

## Input (stdin)

```
N
row_0 : N integers in {-1, 0, +1}
...
row_{N-1} : N integers in {-1, 0, +1}
```

A value of `+1` or `-1` is an **already-excavated** (fixed) cell; a value of `0`
is a **buried** (free) cell you must fill.

## Output (stdout)

`N` lines of `N` integers each, every entry in `{-1, +1}`: your completed
orientation matrix. Fixed cells **must** be reproduced with their given value.

## Feasibility

The output must contain exactly `N*N` entries, each equal to `+1` or `-1`, and
every already-excavated cell must match the input. Any violation scores `0`.

## Objective (maximize)

`|det(M)|`, the exact integer determinant of your completed matrix, computed by
the checker via fraction-free (Bareiss) elimination — no floating-point
tolerance.

## Scoring

Let `B` be the determinant of the checker's own **minimal baseline** completion
(a `+1`-on/below-diagonal, `-1`-above triangular sign matrix with the excavated
cells overwritten), and let `H = N^(N/2)` be the Hadamard upper bound (an
*unreachable* ceiling for odd `N`). With `L = log|det(M)|`:

```
p     = (L - log|B|) / (log H - log|B|)
Ratio = clamp(0.10 + 0.90 * p , 0, 1)
```

Reproducing the baseline yields `Ratio = 0.10`. Because `H` is unreachable, no
layout can reach `1.0`; there is always headroom above the best known
construction. The reported score is `Ratio`.

## Constraints

- `9 <= N <= 27`, `N` odd.
- Roughly 60% of cells are pre-excavated (fixed); you complete the rest.
- The baseline completion is guaranteed non-singular.

## Example (worked score)

Suppose `N = 9`, the baseline determinant is `|B| = 512`, and your completed
grid has `|det(M)| = 4096`. Then `log|B| = 6.238`, `L = 8.318`,
`log H = 4.5 * log 9 = 9.888`, so `p = (8.318 - 6.238)/(9.888 - 6.238) = 0.570`
and `Ratio = 0.10 + 0.90 * 0.570 = 0.613`.
