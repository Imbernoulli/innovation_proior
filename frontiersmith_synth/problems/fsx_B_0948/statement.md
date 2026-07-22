# Die-Bending Springback

A metal-forming lab bends sheets over cylindrical dies. When the punch
releases, the sheet always relaxes back a little from the bend angle it was
forced into — this is **springback**, measured as a single non-negative
number `S`. Each experiment is described by the **die radius** `r` and the
**sheet thickness** `t`; you get measurements from one production
alloy/temper and must predict `S` from `(r, t)` for a **different batch you
never see**.

The catch: the training batch was recorded on **thin** sheets. The grading
batch, from the same alloy, was recorded on **thick** sheets. Metal does not
"remember" a hard bend the same way at every thickness: at some ratio of die
radius to thickness the sheet stops behaving elastically and starts
behaving plastically-saturated instead, and *where that changeover sits
depends on the thickness itself*. On thin sheets most training points sit on
one side of that changeover, with a visible minority already on the other;
on thicker grading sheets the changeover has moved far enough that the OTHER
side is now the majority — but neither split is pure. A model that only
learned to curve-fit `r/t` on the training batch has no way to know that the
same `r/t` value can mean something different once `t` moves into a new
range.

## Input (stdin)

```
n c
r[0]  t[0]  S[0]
r[1]  t[1]  S[1]
...
r[n-1] t[n-1] S[n-1]
```

`c` is the test id; `n` training rows follow (floats, `r,t > 0`). The
held-out grading batch (thicker sheets, same alloy) is **not** given to you.

## Output (stdout): a closed-form expression

Print **one line** containing a single arithmetic expression over the two
variables `r` and `t`, using `+ - * /`, the power operator `**`, parentheses,
numeric constants, and the unary functions `step` (1 if arg>0 else 0), `sig`
(logistic), `relu`, `absv`, `sqrt`. No other names, statements, or lines are
allowed. Example (**illustrative FORM only — not the hidden law**):

```
0.2 + 0.05 * sqrt(r) - 0.03 * t + step(r - 2*t)
```

This only shows the syntax; the real relationship between `(r,t)` and `S` is
different and must be discovered from the data.

## Feasibility

The expression must parse under the grammar above (known names/functions
only, `≤ 60` expression nodes, each numeric constant finite with magnitude
`≤ 1e6`). Evaluating it on any grading row must produce a finite real number
with magnitude `≤ 1e9`; any parse failure, unknown name, division by zero,
domain error, non-finite or out-of-range value, or oversized program scores
`0`.

## Objective (minimise)

Let `MSE` be the mean squared error of your expression against the true
held-out `S`, and `nodes` the number of expression-tree nodes you used. The
grader forms

```
F = MSE * (1 + LAMBDA * nodes)
B = MSE_of_constant_1.2 * (1 + LAMBDA * 1)      # internal baseline
Ratio = min(1000, 100 * B / F) / 1000
```

with a small fixed `LAMBDA`. A constant predictor reproduces `B` (Ratio ≈
0.1); lowering held-out error raises the score, with a mild parsimony tax
against needlessly large expressions. Report the highest Ratio you can.

## Worked score (illustrative numbers only)

Suppose a candidate expression achieves `MSE = 0.04` on the held-out rows
using `nodes = 20`, the constant-1.2 baseline achieves `MSE = 0.90`, and
`LAMBDA = 0.004`. Then `B = 0.90*1.004 = 0.9036`, `F = 0.04*1.08 = 0.0432`,
so `Ratio = min(1000, 100*0.9036/0.0432)/1000 = 1.0` (capped) — a well-fit,
parsimonious expression can reach the cap; a poorly-fit one stays near `0.1`.

## Why the thin-sheet batch is a trap

On the training batch, `S` looks like it lies close to a single smooth curve
of `r/t` — most training rows sit in the same regime, and a smooth fit can
absorb the rest as noise. Any function of `r/t` alone that fits that curve
well is, by construction, blind to `t`'s second role: shifting *where* the
regime changes. On the held-out (thicker-sheet) batch the changeover has
moved, so many `r/t` values that were elastic in training now belong to the
plastic regime — but not all: the held-out batch is a genuine mix of both
regimes, not a clean switch to one side. Neither "assume the training curve
everywhere" nor "assume the other regime everywhere" survives that mix; only
tracking the boundary's own dependence on `t` does.

## Constraints

Time limit 5 s, memory 512 MB. `n` is a few hundred rows. Scoring is fully
deterministic.
