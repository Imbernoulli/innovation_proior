# Shape-Memory Strip

A shape-memory alloy strip is stretched and released along one elongation
path. A logger records, at each tick, the elongation `x` (a normalised
reading) and the strip's restoring force `y`. The strip **remembers which
way you came**: the same elongation `x` produces a different force
depending on whether the strip is currently being stretched further
(loading) or let back in (unloading). Your job: predict the force from the
elongation on a path you have never seen.

## The branch state (fixed rule — not something to discover)

At every tick the strip is on one of two branches, encoded as `b ∈ {+1,-1}`:

- `b[0] = +1` (the strip starts on the loading branch).
- For `i ≥ 1`: `b[i] = +1` if `x[i] > x[i-1]` (elongation increased —
  loading), `b[i] = -1` if `x[i] < x[i-1]` (unloading), and `b[i] = b[i-1]`
  on an exact tie.

This switching rule depends **only on the sign of the change**, never on its
size or on how many ticks separate two reversals — the same physical path
sampled fast or slow, coarsely or finely, produces the same branch sequence.

## Input (stdin)

```
n t
x[0]  y[0]
x[1]  y[1]
...
x[n-1] y[n-1]
```

`t` is the test id; `n` training rows follow (floats). The rows come from a
path with a FEW reversals and a moderate elongation range. The held-out
grading path — used only for scoring — is a DIFFERENT, more agitated path
for the SAME strip: more reversals, a different sampling rate, and often a
wider elongation range. It is never shown to you.

## Output (stdout): one closed-form expression

Print exactly one line: a single arithmetic expression over `+ - * /`,
integer powers `**k` with `|k| ≤ 4`, parentheses, numeric constants, the two
variables `x` and `b`, and the unary functions `sin`, `cos`, `tanh`, `exp`,
`absv`. You do **not** compute `b` yourself — at grading time the checker
derives `b` from the held-out `x` sequence via the rule above and supplies
its value at every tick your expression is evaluated at, exactly like `x`.

**Illustrative FORM only — NOT the hidden law:**
```
0.3 * sin ( x ) + 0.1 * b * exp ( x )
```
This just shows the syntax; the real law is a different shape and you must
discover it from the training data.

## Feasibility

The expression must parse under the grammar above (only the listed
names/functions, `≤ 50` expression nodes, finite constants, powers bounded by
`4`). Any violation, or any non-finite value produced while evaluating your
expression on the held-out path, scores `0`.

## Objective (maximize)

Let `MSE` be the mean squared error of your expression's predictions against
the true held-out force, evaluated with the branch values supplied per the
rule above. The grader forms its own constant-predictor baseline (the mean
held-out force) with baseline error `MSE0`, sets a scale `S = MSE0 / K` with
fixed `K`, and scores

```
F = 1 / (1 + MSE / S)
B = 1 / (1 + K)
Ratio = min(1000, 100 * F / B) / 1000
```

A constant reproduces the baseline exactly (Ratio = 0.1); lower held-out
`MSE` raises the score, capped below `1` even as `MSE -> 0` (sensor noise
keeps the true minimum-achievable `MSE` above zero, so nothing saturates the
scale). Report the highest Ratio you can.

## Why a function of `x` alone is a trap

On the training path the strip loads then unloads at least once, so the
same elongation region is visited by BOTH branches at different forces — a
plain regression `y = f(x)` is structurally unable to fit both. Any such fit
carries a floor error equal to roughly the average branch gap, and that
floor gets *worse*, not better, on the held-out path, which revisits each
region across many more direction reversals at a different sampling rate.
Only a model that explicitly uses `b` — the branch you are currently on —
can close that gap.

## Constraints

Time limit 5 s, memory 512 MB. `n` is at most a few hundred rows, held-out
traces are of comparable size. Scoring is fully deterministic.
