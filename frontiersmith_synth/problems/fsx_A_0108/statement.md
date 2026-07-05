# Geothermal Field Coupling: Minimal Separable Operating Modes

## Problem
A geothermal well field is operated as a set of `n` injection wells, `n` production
wells and `n` depth strata. Reservoir simulation produces a fixed integer
**thermal-coupling tensor** `H`, where

```
H[i][j][k]
```

is the (scaled, integer) heat-transfer coupling between injection well `i`, production
well `j` and stratum `k`.

To drive the field, the control system schedules **separable operating modes**. A single
mode `r` is described by three profiles — an injection profile `u_r` (length `n`), a
production profile `v_r` (length `n`) and a depth profile `w_r` (length `n`) — and it
contributes the rank-1 coupling

```
u_r[i] * v_r[j] * w_r[k]
```

to every triple `(i,j,k)`. Running the whole field is the superposition of its modes.
Each mode consumes exactly **one independent actuator channel** (the per-index products
are formed by fixed valve/pump routing, not extra actuators). Realize the full coupling
tensor `H` with **as few modes as possible** — i.e. find a low-rank CP decomposition of
`H`.

The tensor is engineered to have low *multilinear* rank but an unknown, genuinely small
*CP* rank: cheap per-slice constructions are far from optimal, and no polynomial-time
method is known to certify the minimum. There is no reachable known optimum — better
schedules are found by search.

## Input (stdin)
```
a b c
```
with `a == b == c == n`. Then `a*b` lines follow; the line at position `i*b + j`
(0-based, `i` outer, `j` inner) lists the `c` integers
`H[i][j][0] H[i][j][1] ... H[i][j][c-1]`.

## Output (stdout)
```
R
```
followed by `R` modes. Each mode is written as `a + b + c` (= `3n`) rational numbers,
whitespace separated (you may split across lines freely):

```
u[0] ... u[a-1]  v[0] ... v[b-1]  w[0] ... w[c-1]
```

Each number may be an integer (`-3`), a decimal (`0.5`, `-1.25`) or an exact fraction
(`3/2`, `-7/4`). The tokens `nan`/`inf` are **rejected**. You must output between `1`
and `a*b*c` modes.

## Feasibility
Let `Hhat[i][j][k] = sum_{r=1}^{R} u_r[i] * v_r[j] * w_r[k]`, computed in exact rational
arithmetic. The output is feasible **iff** `Hhat == H` exactly at every entry. Any parse
error, wrong token count, non-finite value, `R < 1`, `R > a*b*c`, or reconstruction
mismatch scores `0`.

## Objective
Minimize `R`, the number of separable operating modes (scalar multipliers).

## Scoring
Let `B` be the checker's internal baseline: the number of nonzero mode-3 fibers `(i,j)`
(i.e. the "one mode per active injection/production pair" construction). For a feasible
output the score is

```
sc    = min(1000, 100 * B / R)
Ratio = sc / 1000
```

So reproducing the baseline (`R == B`) scores `Ratio = 0.1`, and a 10x reduction caps at
`Ratio = 1.0`. Infeasible output scores `Ratio = 0.0`.

## Constraints
- `6 <= n <= 9`, integer entries.
- Exact rational arithmetic throughout; scoring is fully deterministic.

## Example (worked score)
Suppose `n = 6` and the tensor has `B = 36` nonzero fibers.
- The per-fiber baseline uses `R = 36` modes → `Ratio = 100*36/36/1000 = 0.100`.
- A fixed-axis slice factorization uses `R = 18` modes → `Ratio = 100*36/18/1000 = 0.200`.
- A best-of-three-axes multilinear compression uses `R = 12` modes →
  `Ratio = 100*36/12/1000 = 0.300`.
Fewer modes is always better, up to the `10x` cap.
