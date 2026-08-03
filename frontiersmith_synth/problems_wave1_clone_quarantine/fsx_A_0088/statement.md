# Beamforming Gain Tensor: Minimal-Multiplier Combiner Network

## Problem
A satellite ground station operates a phased signal-combining backend. For every triple
`(station antenna i, frequency channel j, time slot k)` the backend must apply a fixed
scalar gain `G[i][j][k]` (an integer, positive or negative, in a small range) to the
incoming sample stream.

Implementing all gains as independent hardware multipliers is wasteful. The backend is
instead built from **combiner stages**. A single stage `r` is described by three weight
vectors — an antenna profile `u_r` (length `a`), a channel profile `v_r` (length `b`) and
a time profile `w_r` (length `c`) — and it contributes the *separable* gain

```
u_r[i] * v_r[j] * w_r[k]
```

to every `(i,j,k)`. The full network is the sum of its stages. Each stage costs exactly
**one scalar multiplier** in the analog fabric (the per-index products are formed by fixed
routing, not by additional multipliers). Your job is to realize the gain tensor `G` with
**as few stages as possible** — i.e. to find a low-rank CP decomposition of `G`.

## Input (stdin)
```
a b c
```
then `a*b` lines, each with `c` integers. The line at position `i*b + j` (0-based, `i`
outer, `j` inner) lists `G[i][j][0] G[i][j][1] ... G[i][j][c-1]`.

Dimensions satisfy `1 <= a,b,c <= 5`.

## Output (stdout)
```
R
```
followed by `R` stages. Each stage is written as `a + b + c` rational numbers (whitespace
separated; you may split across lines freely):

```
u[0] ... u[a-1]  v[0] ... v[b-1]  w[0] ... w[c-1]
```

Each number may be an integer (`-3`), a decimal (`0.5`, `-1.25`) or an exact fraction
(`3/2`, `-7/4`). Scientific notation (`1e3`) and the tokens `nan`/`inf` are **rejected**.

## Feasibility
Let `Ghat[i][j][k] = sum_{r=1}^{R} u_r[i] * v_r[j] * w_r[k]`, computed in exact rational
arithmetic. The output is feasible iff `Ghat == G` **exactly** at every entry. Any parse
error, wrong token count, non-finite value, `R < 1`, or reconstruction mismatch scores 0.

## Objective
Minimize `R`, the number of combiner stages (scalar multipliers).

## Scoring
Let `B` be the number of nonzero entries of `G` (the naive "one stage per nonzero gain"
construction always achieves rank `B`). With `R` your stage count,

```
Ratio = min(1, 0.1 * B / R)
```

The naive per-entry construction scores `0.1`. Halving the multiplier count doubles the
ratio; reaching a tenth of `B` caps at `1.0`. The true minimal rank is unknown and lies
well below what simple axis-slicing achieves, so headroom remains.

## Constraints
- `1 <= a,b,c <= 5`; gains are integers with `|G[i][j][k]| <= 60`.
- `1 <= R <= 1000`.
- Deterministic exact-rational scoring; no timing.

## Example
Suppose `a=b=c=2` and `G` has `B = 6` nonzero entries. A submission with `R = 3` stages
that reconstructs `G` exactly scores `min(1, 0.1*6/3) = 0.2`. The per-entry baseline
(`R = 6`) would score `0.1`.
