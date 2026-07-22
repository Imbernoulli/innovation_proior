# Backward Heat Sourcing: Reaching a Temperature Snapshot

## Problem
A square `N x N` plate cools by discrete heat diffusion on a **periodic** (wrap-around)
grid. One diffusion step maps a field `u` to
`u'[i][j] = u[i][j] + alpha * (u[i+1][j] + u[i-1][j] + u[i][j+1] + u[i][j-1] - 4*u[i][j])`
(indices mod `N`). Applying this step `T` times is the forward operator `F`.

You are given a **target** field `y` (a temperature snapshot). Choose an **initial source
field** `s` with `s[i][j] >= 0` everywhere. Your plate is heated to `s`, then diffused `T`
steps, producing `F(s)`. You are scored on how close `F(s)` gets to `y`, minus a charge for
using many separate sources.

## Input (stdin)
```
N T alpha cost
```
then `N` lines, each with `N` real numbers: the target field `y` (row-major).
`N <= 32`, `2 <= T <= 12`, `0 < alpha <= 1/8`, `cost > 0`. `y` may contain negative values.

## Output (stdout)
`N` lines, each `N` real numbers: your source field `s`. Every `s[i][j]` must be a finite
number with `s[i][j] >= 0`.

## Feasibility
Output must parse as exactly `N*N` finite reals, all `>= 0`. Any negative, `nan`, `inf`,
missing or extra value makes the submission infeasible (score 0).

## Objective (minimize)
Let `nz` = number of cells with `s[i][j] > 1e-6` (the count of active sources). The cost is
```
J(s) = sum over all cells of ( F(s)[i][j] - y[i][j] )^2   +   cost * nz .
```
Lower `J` is better. The score compares `J(s)` to the do-nothing baseline `B = J(0) =
sum of y[i][j]^2` (all-zero sources): `Ratio = min(1, 0.1 * B / J(s))`. All-zero scores
`0.1`; driving `J` to one tenth of `B` reaches `1.0`.

## Why this is hard (the trap)
`F` is a **low-pass filter**: high-frequency detail is multiplied by a factor as small as
`(1-8*alpha)^T` per mode and is essentially **unreachable** — no non-negative source can
reproduce it, and `y`'s negative regions are unreachable outright (a non-negative source
diffuses to a non-negative field). The textbook move — *invert* `F` (backward-diffuse `y`,
i.e. deconvolve the heat kernel) — is ill-posed: it amplifies those tiny high-frequency
factors, producing wild oscillations and **negative** source values that are infeasible.
The other obvious move — set each source equal to the temperature you want there
(`s = y` at the hot cells) — badly **undershoots**, because one unit of source diffuses to a
peak far below `1`, so `F(s)` never rises near `y`.

## Constraints
Deterministic scoring. Time limit 5 s, memory 512 MB. `N <= 32`.

## Example (illustrative, small)
Suppose two nearby cells in `y` are hot. Placing a single well-scaled source between them —
its diffused blob covering both — can match the low-frequency shape of that region with **one**
active cell, beating both a dense inversion (huge `nz`, oscillating and infeasible) and the
undershooting `s = y` copy. The winning idea: **work only in the reachable low-frequency
subspace** (low-pass `y` before you fit), and **spend a few well-placed, correctly-scaled
sources** rather than one per pixel. The exact `alpha`, `cost` and `T` come from the input, so
you must read them and balance fit against the per-source charge.
