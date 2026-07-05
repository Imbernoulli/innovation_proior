# Watchtower Alert Kernel: Minimal Rank Decomposition

## Problem

A ring of forest-fire **watchtowers** shares a calibrated early-warning model. The
model is a 3D integer **alert tensor** `T` of shape `a x b x c`, indexed by

- `i` — watchtower (`0..a-1`),
- `j` — wind sector (`0..b-1`),
- `k` — fuel-moisture band (`0..c-1`).

`T[i][j][k]` is the (exact, integer) ignition-alert weight for that combination.

To run on the towers' low-power radios, the model must be re-expressed as a sum of
`R` **rank-1 alert primitives**

```
T[i][j][k] = sum_{r=1}^{R}  u_r[i] * v_r[j] * w_r[k]
```

where each primitive `r` is an outer product of a tower-profile `u_r` (length `a`),
a sector-profile `v_r` (length `b`) and a moisture-profile `w_r` (length `c`), with
**rational** entries. Each primitive costs one scalar multiply per query, so **fewer
primitives = fewer multiplies = longer battery life**. Reproduce `T` **exactly** with
as few primitives as possible.

## Input (stdin)

```
a b c
```

followed by `a*b` lines: for each `i` in `0..a-1` and `j` in `0..b-1` (in that order),
one line of `c` integers giving the fiber `T[i][j][0..c-1]`.

Dimensions satisfy `a = b = c = n` with `4 <= n <= 7`.

## Output (stdout)

```
R
```

then `R` lines, each with `a + b + c` rationals: the concatenation
`u_r[0..a-1]  v_r[0..b-1]  w_r[0..c-1]`. Each rational is written either as an integer
`p` or as a fraction `p/q` (e.g. `-3`, `0`, `5/2`, `-7/4`). No decimals, no exponents,
no `nan`/`inf`.

## Feasibility

The decomposition must reconstruct `T` **exactly** under rational arithmetic:
`sum_r u_r[i] v_r[j] w_r[k] == T[i][j][k]` for every `(i,j,k)`. Any mismatch,
malformed token, non-finite value, or wrong token count scores `0`.

## Objective

**Minimize** `R`, the number of rank-1 primitives (scalar-multiply count).

## Scoring

Let `B` be the number of nonzero mode-3 fibers of `T` (the checker's internal
baseline: one primitive `e_i (x) e_j (x) T[i][j][:]` per nonzero fiber reconstructs `T`
exactly). Your score is

```
Ratio = min(1.0, 0.1 * B / R)
```

So the fiber baseline scores `~0.1`; halving the primitive count roughly doubles the
ratio. The true minimal rank is **not known** (tensor rank is NP-hard, and the planted
structure only bounds it) — there is no closed-form optimum to reach.

## Constraints

- `4 <= n <= 7`, `a = b = c = n`.
- Exact rational arithmetic only; deterministic scoring.
- `R` at most `5*a*b*c`.

## Example (worked score)

Suppose `n = 4`, all 16 mode-3 fibers are nonzero, so `B = 16`. The fiber baseline
uses `R = 16` primitives → `Ratio = 0.1 * 16 / 16 = 0.1`. If instead you factor the
tensor's frontal slices and reach `R = 8` primitives → `Ratio = 0.1 * 16 / 8 = 0.2`.
Reaching `R = 4` (if achievable) would give `Ratio = 0.4`.
