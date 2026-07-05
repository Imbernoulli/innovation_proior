# Neural Scaling Law with an Irreducible Floor

## Problem
A pretraining team has swept a bank of **small exploratory LLM runs**, each
recording the training compute `C`, the number of training tokens `D`, and the
achieved validation cross-entropy loss `L`. The loss follows a fixed but
**unknown** neural scaling law with an *irreducible entropy floor* `E > 0` — a
loss no amount of compute or data can beat. Every logged loss is corrupted by
run-to-run measurement noise.

The team must decide how to spend its next, **much larger** training budget, so
it needs a compact analytic law that still predicts loss correctly at compute
and token counts **far beyond** the exploratory sweep. Your job is to recover a
closed-form law `L(C, D)` that **generalises to that large-scale extrapolation
region**, not one that merely memorises the noisy small-run points.

Each test id corresponds to a different model family (a different hidden law).

## Input (stdin)
```
line 1:            n_train   test_id
next n_train lines: C  D  L        (space-separated floats)
```
`C` and `D` are in normalised units (small exploratory runs: `C` up to ~80,
`D` up to ~50). `test_id` is provided for reference only; the law must be
inferred from the data.

## Output (stdout)
A **single line** holding a Python expression for `L` in the variables `C` and
`D`. Allowed operators: `+ - * / ** %`; allowed functions: `exp, log, sin, cos,
sqrt, tanh, abs`; numeric literals are allowed. No other names, attributes,
calls, or imports are permitted.

Example output line (illustrative FORM only — **not** the hidden law):
```
2.0 + 3.5*exp(-0.1*C) - 0.2*log(D)
```

## Feasibility
The output must be exactly one line, parse as an expression over the allowed
grammar, and evaluate to a finite real number on every held-out point.
Anything else scores `0`.

## Objective (minimise held-out error, complexity-penalised)
The grader deterministically regenerates a **held-out extrapolation split** at
**much larger compute and data** (`C` in `[200, 4000]`, `D` in `[80, 500]` —
well beyond the training box) plus irreducible measurement noise, then evaluates
your expression there. Let `heldout_MSE` be the mean squared error and
`complexity` the node count of your expression. With `LAMBDA = 0.003`:

```
F = heldout_MSE * (1 + LAMBDA * complexity)
B = baseline_MSE * (1 + LAMBDA * 1)        # baseline = constant train mean
Ratio = min(1000, 100 * B / F) / 1000
```

## Scoring
A constant prediction reproduces the baseline (`Ratio ≈ 0.1`). Recovering the
power-law shape — including the irreducible floor `E` — drives held-out error
toward the noise floor and raises the ratio, but the irreducible noise floor and
the fact that the scaling exponents are hidden keep even a strong recovery below
`1.0`. Simpler expressions with the same error are rewarded via the complexity
term. The per-test score is `Ratio`; the final score averages over the
difficulty ladder.

## Constraints
- `test_id` in `1..10`; `n_train` between 48 and 120 (shrinks with difficulty).
- Measurement noise grows with the test id.
- Expression output ≤ 200000 bytes, single line.

## Example (worked score)
Suppose on some test your expression yields `heldout_MSE = 0.30` with
`complexity = 15`, while the constant baseline has `baseline_MSE = 3.0`. Then
`F = 0.30*(1+0.003*15) = 0.3135`, `B = 3.0*1.003 = 3.009`,
`Ratio = min(1000, 100*3.009/0.3135)/1000 = 0.960`... capped by the noise floor
in practice. A plain constant gives `F ≈ B` and `Ratio ≈ 0.1`.
