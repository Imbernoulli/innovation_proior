# Substation Phasor Coupling: Fewest Multiplier Primitives

## Problem
A substation controller runs a real-time state estimator over a three-way **phasor
coupling tensor**

    T[i][j][k] = joint coupling coefficient among bus i, feeder line j and harmonic band k.

The controller cannot evaluate `T` directly; it can only chain **rank-1 multiplier
primitives**. A primitive is a triple of weight vectors `(a, b, c)` and it contributes

    a . busVec  *  b . lineVec  *  c . harmVec

to the estimate. Each primitive costs **exactly one hardware scalar multiply**. A
program of `R` primitives realizes the tensor iff

    T[i][j][k] = sum_{r=1..R} a_r[i] * b_r[j] * c_r[k]      for all i, j, k.

Your job: realize the shipped tensor exactly with **as few primitives as possible**.
`R` is the tensor rank you achieve; fewer multiplies is a cheaper controller.

The tensor is planted so that every harmonic-band slice is a low-rank bus×line matrix,
but the total number of planted primitives *exceeds every dimension* (over-complete
rank). Polynomial diagonalization methods therefore cannot recover the true optimum, and
the genuine minimum is unknown — this is an open-ended optimization, not a puzzle with a
posted answer.

## Input (stdin)
```
B L H
```
then `H` harmonic-band slices; slice `k` is `B` lines of `L` integers, where the value on
bus-row `i`, line-column `j` is `T[i][j][k]`. All entries are integers.

## Output (stdout)
```
R
```
then `R` primitives. Each primitive is **3 whitespace-separated lines**:
```
a_1 a_2 ... a_B        # bus weights      (B rationals)
b_1 b_2 ... b_L        # line weights     (L rationals)
c_1 c_2 ... c_H        # harmonic weights (H rationals)
```
Each weight may be an integer, a decimal, or an exact fraction `p/q`. `NaN`/`Inf` are
rejected. Whitespace/newlines are free-form; the checker reads `1 + R*(B+L+H)` tokens.

## Feasibility
The reconstruction `sum_r a_r[i]*b_r[j]*c_r[k]` must equal `T[i][j][k]` for **every**
`(i,j,k)`, checked in **exact rational arithmetic**. Any mismatch, malformed schema,
wrong token count, `R < 1`, absurdly large `R`, or non-finite weight scores `0`.

## Objective
Minimize `R`, the number of scalar multiplies.

## Scoring
Let `B0` = number of non-zero harmonic (mode-3) fibers — a trivial feasible rank the
checker builds itself. With `F = R`:

    Ratio = min(1000, 100 * B0 / F) / 1000

A trivial one-primitive-per-fiber program gives `R = B0` -> `Ratio = 0.1`. Halving the
primitive count doubles the ratio; reaching one tenth of the baseline caps at `1.0`.

## Constraints
- `4 <= B, L <= 7`, `2 <= H <= 4`.
- Integer tensor entries.
- Deterministic, exact scoring — no randomness, no timing.

## Example (worked score)
Suppose `B0 = 25` non-zero harmonic fibers. A submission using `R = 6` primitives scores

    Ratio = min(1000, 100 * 25 / 6) / 1000 = 416.67 / 1000 = 0.4167,

while the trivial `R = 25` program scores `0.1`. (Numbers illustrative — the shipped
instances differ.)
