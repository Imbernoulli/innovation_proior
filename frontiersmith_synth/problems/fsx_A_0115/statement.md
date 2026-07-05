# Watchtower Overlap Constant

## Problem
A long mountain ridge is divided into `n` equal cells. Along the ridge you deploy a
**surveillance profile** — a non‑negative integer intensity `f[0..n-1]` per cell (number of
sensors / staffed watchtowers assigned to that cell). At least one cell must be positive.

The regional fire office worries about **paired‑coverage concentration**. For a combined
position `j` (0 ≤ j ≤ 2n-2), the *paired coverage at j* is the self‑convolution

```
g[j] = sum over i of f[i] * f[j-i]      (self-convolution f*f of the profile)
```

`g[j]` measures how much coverage is contributed by all cell pairs `(i, j-i)` whose indices
sum to `j` — the total duplicated effort focused on combined position `j`. If any single `j`
dominates, the deployment is fragile: it concentrates paired coverage at one combined scale and
leaves others thin.

You must spread the effort so that **no lag dominates**, in the scale‑free sense of the
first autocorrelation inequality. Define the **overlap constant**

```
c = 2*n * max_j g[j] / (sum_i f[i])^2
```

`c` is invariant to scaling the whole profile. Minimize it.

## Input (stdin)
One line with two integers:
```
n U
```
`n` = number of ridge cells, `U` = maximum intensity allowed per cell.

## Output (stdout)
`n` integers `f[0] f[1] ... f[n-1]` (whitespace‑separated, on one or more lines), each in
`[0, U]`, with `sum_i f[i] > 0`.

## Feasibility
Exactly `n` integers, every value in `[0, U]`, and total sum strictly positive. Any violation
scores `0`.

## Objective
Minimize the overlap constant `c = 2*n * max_j g[j] / (sum f)^2`, where the maximum is over
all combined positions `j = 0 .. 2n-2` and `g[j] = sum_i f[i]*f[j-i]` (self-convolution).

## Scoring
The checker builds the flat baseline `f = (1,1,...,1)`, which always gives `c = 2`. Your score
rewards how far below that baseline you push `c`:

```
val   = (sum f)^2 / (n * max_j g[j])         # equals 2 / c
Ratio = min(1.0, 0.1 * val^6)
```

The flat profile scores `Ratio = 0.1`. Driving `c` toward the packing‑theoretic floor near
`~1.5` raises the score steeply; `Ratio = 1.0` (which needs `c < 1.36`) is provably out of
reach, so the problem is genuinely open‑ended — there is no attainable optimum, only better
constructions.

## Constraints
- `8 <= n <= 32`, `U = 1000`.
- Scoring is exact integer arithmetic on `g` and the sum; fully deterministic.

## Example
For `n = 8`, `U = 1000`, the flat profile `1 1 1 1 1 1 1 1` has `sum = 8` and self‑convolution
`g = (1,2,3,4,5,6,7,8,7,6,5,4,3,2,1)`, so `max g = 8`, `c = 2*8*8/64 = 2.0`,
`val = 64/(8*8) = 1.0`, `Ratio = 0.1`. A hand‑tuned uneven profile that lowers `max g` relative
to `(sum f)^2` scores strictly higher.
