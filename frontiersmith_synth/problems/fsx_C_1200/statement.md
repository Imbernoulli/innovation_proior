# Churn Hazard Forecast

A subscription business logs customers over a **visible observation window**
of length `T_obs`. Each customer belongs to a cohort with a continuous
loyalty covariate `x` in `[0,1]` (six fixed buckets: `0.0, 0.2, ..., 1.0`).
Each customer's true (hidden) tenure is drawn from a hazard whose **shape**
and **scale** both depend on `x`: some cohorts have a **decreasing** hazard
(a long, sticky survivor tail), others a **rising** one (churn bursts around
a typical tenure).

Customers who have not yet churned when the window closes are
**right-censored**: you only learn they were still active at `T_obs`, not
their true exit time. Naively averaging "observed tenure" (treating the
censoring cutoff as if it were a real exit) is therefore biased **short** —
and the bias is worst for exactly the long-tenured cohorts you most want to
understand.

Your job: reverse-engineer, from the censored sample alone, a closed-form
predictor of the survival probability `S(t, x)` = P(a customer of cohort `x`
is still active at tenure `t`), including at horizons **beyond** the visible
window.

## Input (stdin)

```
N  T_obs  testId
x_0  observed_tenure_0  censored_0
x_1  observed_tenure_1  censored_1
...
```

`N` customers follow. `censored_i = 1` means "still active at `T_obs`, true
exit unknown" (`observed_tenure_i = T_obs`); `censored_i = 0` means the true
exit was observed (`observed_tenure_i` < `T_obs`). The held-out grading
horizons — some well beyond `T_obs` — are **not** given to you.

## Output (stdout): a closed-form expression

Print **one line**: a single arithmetic expression over the variables `t`
(tenure) and `x` (cohort covariate), using `+ - * / **`, parentheses,
numeric constants, the constants `pi`/`e`, and the unary/n-ary functions
`exp`, `log`, `sqrt`, `abs`, `min`, `max`. No other names, calls, or syntax
are allowed.

**Illustrative FORM only — NOT the hidden law** (a different function family;
discover the real relationship's shape from the data):

```
exp ( -0.02 * ( t ** 2 ) - 0.10 * x * t )
```

## Feasibility

The expression must parse under the grammar above (known names/functions
only, correct arities, finite constants, ≤ 80 expression nodes). Any
violation, or any non-finite / non-real value produced anywhere on the
held-out grid, scores `0`.

## Objective (maximise accuracy)

The grader regenerates a held-out grid of `(t, x)` points — `t` at six
multiples of `T_obs` from `0.4x` up to a genuinely extrapolated `3.0x`, `x` on
a finer grid than the six training buckets — and computes the true
`S(t, x)` analytically from the (hidden) hazard law. Your submitted
expression is evaluated (and clipped to `[0,1]`) at every grid point; let
`MAE` be the mean absolute error against the true `S`, and `nodes` the
expression's node count. The grader forms

```
F = MAE + 0.002 * max(0, nodes - 60)          # light parsimony tax
B = MAE of the constant predictor 0.5          # internal baseline
Ratio = min(1000, 100 * B / F) / 1000
```

A constant guess reproduces `B` (Ratio ≈ 0.1). Lower held-out error raises
the score; report the highest `Ratio` you can. The held-out label at each
grid point carries a small fixed measurement-noise floor (seeded by the test
id and point index, independent of your submission), so even a perfectly
recovered hazard shape will not drive `MAE` to exactly `0` — there is always
some headroom above any reference solution.

## Why the visible window is a trap

Inside `[0, T_obs]` almost any curve tracking the raw observed-tenure
statistics looks reasonable. But those statistics are censoring-biased, and a
curve fit to them says nothing trustworthy about `t > T_obs` — while grading
horizons run out to `3 * T_obs`. What *is* estimable despite censoring is the
**hazard shape**: the exit rate among customers still at risk at each moment
inside the window. A model recovering the shape (not the biased mean)
extrapolates safely; one that fits the mean cannot.

## Constraints

Time limit 5 s, memory 512 MB. `N` up to 1500. Cohort weights and censoring
severity vary sharply across the 10 tests — some cohorts are heavily
censored, some are a small minority of the sample, some both. Scoring is
fully deterministic.
