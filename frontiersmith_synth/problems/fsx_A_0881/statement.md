# The Smith's Quench Log

A smith quenches many workpieces and logs, for each, a temperature reading
`T` and the one-tick temperature **change** `dT` that followed it — the
metal is already dropping in temperature by the time the shop's hand
thermometer settles on a number, so every logged `T` sits in a modest,
"already-cooling" band; nothing glowing-hot was ever logged.

Physically, the cooling rate is the sum of **two additive regimes**: an
ordinary conduction/convection term, proportional to how far the metal sits
above the shop's ambient temperature, and a radiative term, which grows with
the **cube** of that excess. On the logged range the radiative term is a
minor correction next to conduction and next to the thermometer's own
reading noise — it barely nudges any single row. Your job is to recover a
predictor of `dT` from `T` that works not only on the smith's log but also
on blades quenched from far hotter, forge-glowing starting temperatures the
thermometer could never reach and that were never logged. A model that
merely explains the log well is not enough: you will also be graded on that
much hotter, unseen regime, where a term too small to matter in the log can
come to dominate entirely.

## Input (stdin)

```
n t ambient
T[0] dT[0]
T[1] dT[1]
...
T[n-1] dT[n-1]
```

`t` is the test id, `ambient` the shop's ambient temperature. `n` logged
rows follow (floats). The held-out grading temperatures are a **much
hotter** range never logged; they are not given to you.

## Output (stdout): one arithmetic expression

Print a single line: one arithmetic expression in the variable `T` that
predicts `dT`. Allowed: `+ - * /`, parentheses, numeric constants, and `**`
— but the right-hand side of `**` must be a literal integer between `0` and
`6` (e.g. `(T-20)**3` is fine; `T**k` or `T**2.5` is not). No function calls,
no other names.

**Illustrative FORM only — NOT the hidden law:**

```
0.02 * (120 - T) - 0.5
```

This just shows the syntax; the real law's shape (and whether the cubic
term is even worth including) must be discovered from the data.

## Feasibility

Your line must parse under the grammar above (only `T` and numeric
constants; `**` exponents restricted as above), be at most 200 characters,
and evaluate to a finite number at every point used for grading. Any
violation scores `0`.

## Objective (maximise)

Let `acc(T) = max(0, 1 - |pred(T) - true(T)| / (|true(T)| + 1e-3))` be the
per-point accuracy of your expression against the true hidden `dT(T)`. The
grader draws two deterministic point sets from the hidden law:

- 300 **fresh** points in the same modest range as the log (unseen exact
  values) — weight `0.3`
- 300 **held-out** points from the much hotter, never-logged range — weight
  `0.7`

```
F = 0.3 * mean(acc over fresh-log-range points) + 0.7 * mean(acc over held-out points)
B = the SAME blended metric for the constant predictor "mean of your own
    logged dT column" (the grader's own trivial construction)
Ratio = min(1000, 100 * F / max(1e-9, B)) / 1000
```

A flat constant reproduces `B` (`Ratio ~= 0.1`). Because the held-out weight
dominates, a predictor that fits the log almost perfectly but omits the
radiative regime still scores modestly at best — it only ever "wins" the
30% slice.

## Why the log is a trap

Fit any decent curve to the log alone and you will likely get an excellent
in-sample score: the radiative term is smaller than the thermometer's own
noise on any single row, so a model that ignores it still looks like a
near-perfect fit. The tell is not in any one row's residual — it is in the
**systematic** drift of residuals, averaged across many rows, once you
subtract the obvious dominant term. Trusting training goodness-of-fit alone
misses it; aggregating the pattern across the whole log will not.

## Constraints

Time limit 5 s, memory 512 MB. `n` ranges from a couple thousand to about
thirteen thousand rows across the 10 tests — the radiative term is small
enough that recovering it reliably takes a sizeable log. Scoring is fully
deterministic.
