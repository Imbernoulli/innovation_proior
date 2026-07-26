# Catalytic Assay

A chemist ran a series of short assays on a reaction that consumes a
substrate `S` in the presence of a fixed amount of catalyst `C`. Each assay
reports an observed conversion **rate** for one `(S, C)` pair. The lab log
does not say what kinetic law governs the reaction — first order, saturating,
something else — you must infer it from the data.

Every assay you are given was run at **low substrate concentration** (a
dilute regime), across a handful of different catalyst loadings. You will be
graded on a **held-out** batch of assays that includes substrate
concentrations well beyond anything you were shown, and — for some runs —
catalyst loadings that never appeared in your training log at all. A law that
merely mimics the dilute data numerically, without capturing the reaction's
actual mechanism, will fail badly out there.

## Input (stdin)

```
test_id n_regimes n_pts
S_1 C_1 rate_1
S_2 C_2 rate_2
...
```

`n_regimes` distinct catalyst levels `C` were assayed, `n_pts` substrate
points each, so `n_regimes * n_pts` data rows follow the header. All values
are floats. The held-out grading batch (different `S`, and possibly different
`C`, for the same hidden reaction) is **not** shown to you.

## Output (stdout)

Print **one line**: a closed-form algebraic expression for `rate` as a
function of the two variables `S` and `C`, using only numeric constants and
the operators `+ - * / **` with parentheses. No function calls, no other
variable names.

**Illustrative FORM only — NOT the hidden law** (just shows the required
syntax; it is unrelated to reaction kinetics):

```
2.5 * (S - 3.0) ** 2 + 1.2 * C
```

The real reaction's rate law has a different shape, which you must discover
from the data — the worked syntax above uses a polynomial in `S` purely to
illustrate the grammar.

## Feasibility

Your line must parse as a valid expression over `S`, `C`, numeric constants
and `+ - * / **` (≤ 300 characters, ≤ 40 expression nodes). It must evaluate
to a finite number at every held-out `(S, C)` pair — a parse error, an
unknown name, a division by zero, or a non-finite result at any point scores
`0`.

## Objective (minimise)

Let `MSE` be the mean of the squared *relative* error of your expression's
predictions against the true held-out rates (`(pred - true) / (Vmax * C)`,
where `Vmax` is the reaction's hidden maximum rate — a fixed, unknown
normaliser; each term is clipped to keep any single wild point from
dominating), and `nodes` the number of expression nodes you used. The grader
forms:

```
F = MSE * (1 + LAMBDA * nodes)
B = MSE_of_the_mean_training_rate * (1 + LAMBDA)      # the grader's own baseline
Ratio = min(1000, 100 * B / F) / 1000
```

with a small fixed `LAMBDA`. Predicting a constant reproduces `B` (Ratio ≈
0.1). The more your expression's *shape* — not just its fit to the training
rows — matches the reaction's actual mechanism, the lower `F` gets and the
higher your score. Assay measurement noise keeps even a very good law off the
ceiling.

## Why the dilute log is a trap

At low `S`, almost any smooth kinetic law looks locally linear in `S`, so a
straight-line (or bilinear-in-`S,C`) fit can reproduce the training rows to
within noise. That fit's behaviour as `S` grows is decided entirely by its
slope near `S=0` — it has no way to "know" whether the true mechanism keeps
climbing or levels off. The held-out batch is specifically built to expose
that difference: the deeper the extrapolation, the more a law that keeps
extrapolating linearly diverges from a law that saturates. Distinguishing the
two requires either a mechanistic hypothesis or held-out evidence at higher
`S`, not just a better local slope.

## Constraints

`n_regimes` is 2–4, `n_pts` is 6–9, all data floats in a reasonable range.
Time limit 5 s, memory 512 MB. Scoring is fully deterministic.
