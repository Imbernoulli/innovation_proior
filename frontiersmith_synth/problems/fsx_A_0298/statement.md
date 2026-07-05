# Abyssal Crosstalk Tensor: Minimum-Multiplier Repeater Mixing Network

## Problem

A trans-oceanic optical repeater sits on the seabed between **a** inbound fibres,
**b** outbound fibres and **c** DWDM wavelength channels. Laboratory characterisation
of the amplifier board produces an integer **crosstalk coupling tensor**

```
T[i][j][k]   (0 <= i < a, 0 <= j < b, 0 <= k < c)
```

giving the coupling from inbound fibre *i* to outbound fibre *j* on wavelength *k*.

The board realises `T` as a **mixing network** of separable rank-1 *stages*. A stage is
a triple of gain vectors `(u, v, w)` (lengths `a`, `b`, `c`) that contributes the outer
product `u ⊗ v ⊗ w`, i.e. it adds `u[i]·v[j]·w[k]` to channel `(i,j,k)`. Each stage
consumes exactly **one scalar multiplier** on the board. Your job: reproduce `T` exactly
with as **few stages** (multipliers) as possible — a minimum-rank CP decomposition.

## Input (stdin)

```
a b c
```
then `a·b` rows, one per `(i,j)` pair in row-major order (`i` outer, `j` inner), each row
listing the `c` integers `T[i][j][0] … T[i][j][c-1]`.

## Output (stdout)

```
R
```
followed by `R` lines, one per stage. Each stage line lists `a + b + c` rational numbers
(integers, `p/q`, or decimals) — the gain vectors concatenated `u[0..a-1] v[0..b-1] w[0..c-1]`.

## Feasibility

The decomposition is accepted only if the stages reconstruct the target **exactly**:
```
sum_{r=1..R} u_r[i] · v_r[j] · w_r[k]  ==  T[i][j][k]   for all (i,j,k)
```
using exact rational arithmetic. Any shape error, non-rational / non-finite entry, wrong
token count, or a single mismatched coefficient scores **0**.

## Objective

Minimise `R`, the number of stages (scalar multipliers).

## Scoring

Let `B` = number of nonzero coefficients in `T` (the naive one-multiplier-per-entry cost).
With your feasible stage count `R`:
```
Ratio = min(1, 0.1 * B / R)
```
The per-entry decomposition (`R = B`) scores `0.1`; using ten times fewer multipliers
caps the score at `1.0`. The tensor has a **planted overcomplete rank** (larger than any
axis length), so simultaneous-diagonalisation methods cannot recover the optimum and the
true minimal `R` is unknown — slice-rank factorisations only give upper bounds.

## Constraints

- `3 <= a <= b < c <= 8`.
- All `T[i][j][k]` are integers with `|T[i][j][k]|` small (bounded by the planted rank).
- Deterministic scoring; exact rational arithmetic throughout.

## Example (worked score)

For a `3×4×5` tensor with `B = 60` nonzero coefficients: the per-entry baseline uses
`R = 60` stages → `Ratio = 0.1`. Slicing along the wavelength axis and rank-factoring each
`3×4` slice gives `R = 5·3 = 15` → `Ratio = min(1, 0.1·60/15) = 0.4`. Choosing the best of
the three axes reaches `R = 3·4 = 12` → `Ratio = min(1, 0.1·60/12) = 0.5`. Any decomposition
below the slice-rank ceiling scores higher, up to the `1.0` cap.
