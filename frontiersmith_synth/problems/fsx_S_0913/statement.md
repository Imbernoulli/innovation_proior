# Plague Scrolls Counted Only on Market Days

A contagion is spreading through a town whose scribes keep a peculiar
ledger. Every day the true number of new cases grows (or, after a decree,
grows more slowly) — but what gets **inscribed on a scroll** each day is not
that number directly. It is filtered through two administrative facts of the
town that never change: a **market-day rhythm** (case-loads are tallied more
fully on busy market days and thinly on one quiet day of the week) and a
**scribe-capacity ceiling** (on any day, however large the true load, only so
many scrolls can physically be written — a soft saturation).

On day `n` a magistrate's decree takes effect: the contagion's growth rate is
multiplied by a **known** factor `f` (you are told `f`). The market rhythm
and the scribes' capacity are unrelated to the decree — nothing about the
town's bureaucracy changes when transmission does. Your job: predict the
scroll counts for the days *after* the decree, using only what you can infer
about the town from the days *before* it.

## Input (stdin)

```
n tid f
t[0]  reported[0]
t[1]  reported[1]
...
t[n-1] reported[n-1]
```

`n` is the number of training days (all strictly before the decree), `tid`
is the test id, `f` is the known post-decree growth multiplier. Each
remaining row is a day index `t` (0-indexed, day-of-week = `t mod 7`) and the
integer scroll count recorded that day.

## Output (stdout): one arithmetic expression

Print a **single line** containing one Python-style arithmetic expression
over the variables `t` (a day index), `n`, and `f` (exactly the values from
the header, re-bound at grading time). Allowed: `+ - * / %`, parentheses,
numeric constants, a bracketed list of numeric constants with one subscript
`[...][expr]` (for a day-of-week lookup), and the unary functions `exp`,
`log`, `sqrt`, `absv`. At most 140 expression nodes.

**Illustrative FORM only — NOT the hidden law:**

```
3.0 + absv(t - n) * f
```

This just shows the syntax; the real relationship between reported counts,
time, and the decree is different and must be discovered from the data.

## Feasibility

The expression must parse under the grammar above (known names/functions
only, finite constants, size within bounds). Any violation, or any
non-finite value produced while evaluating it on the grading days, scores
`0`.

## Objective (minimise)

Your expression is evaluated at the **held-out** days `t = n .. n+H-1`
(roughly one to two weeks immediately after the decree — a genuine
extrapolation into a regime you never observed). Let `RMSE` be its
root-mean-squared error against the true (noisy) scroll counts there, and
`nodes` the size of your expression. The grader forms

```
F = RMSE * (1 + LAMBDA * nodes)
B = RMSE_of(mean of the last training week) * (1 + LAMBDA * 1)
Ratio = min(1000, 100 * B / F) / 1000
```

with a small fixed `LAMBDA`. A flat continuation of the last training week
reproduces `B` (Ratio ≈ 0.1); lower held-out error raises the score, with a
light parsimony tax against needlessly large expressions.

## Why fitting the raw counts is a trap

Before the decree, a smooth trend fit to the reported counts looks
excellent — the market rhythm looks like noise and the scribe ceiling looks
like the epidemic "naturally" slowing down late in the window. But that
apparent slowdown is administrative, not epidemiological: it is caused by
scrolls running out, not cases. The decree only changes the underlying
transmission rate; it does not touch the rhythm or the ceiling. A predictor
that never separated the two has no principled way to know which part of
its fitted curve the decree should rescale, and blindly rescaling the whole
(already-biased) trend misses badly the moment the regime actually
changes. A predictor that first isolates the growth rate from the
reporting operator — then rescales *only* the growth rate by the known
factor `f`, keeping the rhythm and ceiling fixed — extrapolates correctly.

## Constraints

`n` is 28–38. Time limit 5 s, memory 512 MB. Scoring is fully
deterministic and depends only on the test id.
