# Terminal Clearance Extrapolation

## Problem
In a Phase-I drug trial a compound is given as an intravenous bolus and its
plasma concentration `C` (mg/L) is followed over time `t` (hours). The
disposition obeys a fixed but **unknown** two-compartment (biexponential) decay
law: a fast *distribution* phase followed by a slow *terminal elimination*
phase.

The assay schedule only lets you draw blood in the **early sampling window**
`t ∈ [0.08, 3.0]` hours, and every reading carries multiplicative assay error.
Regulators, however, need the model to predict the **late clearance tail**
(`t` well beyond the sampling window), which is governed almost entirely by the
slow elimination rate. Your task is to recover a compact closed-form law
`C(t)` from the early draws that **extrapolates into that terminal region** —
i.e. one that has correctly separated the slow elimination rate from the early
mixed signal, not one that merely traces the early curve.

Each test id is a **different compound** (a different hidden law).

## Input (stdin)
```
line 1:              n_draws   test_id
next n_draws lines:  t   C          (space-separated floats)
```
`test_id` is provided for reference only; the law must be inferred from the data.

## Output (stdout)
A **single line** holding a Python expression for `C` in the variable `t`.
Allowed operators: `+ - * / ** %`; allowed functions:
`exp, log, sin, cos, sqrt, tanh, abs`; numeric literals are allowed. No other
names, attributes, calls, or imports are permitted.

Example output line (illustrative **FORM only — not** the hidden law):
```
3.0*exp(-0.4*t) + 0.7
```

## Feasibility
The output must be exactly one line, parse as an expression over the allowed
grammar, and evaluate to a finite real number on every held-out point.
Anything else (empty, multi-line, unknown names, `nan`/`inf`) scores `0`.

## Objective (minimise held-out log-concentration error, complexity-penalised)
The grader deterministically regenerates a **held-out late clearance tail**
(`t ∈ [4, 10]` h — a different, non-overlapping region from the training draws)
with irreducible measurement noise, then scores your expression there on the
**log-concentration** scale (the standard PK metric, which handles the wide
dynamic range). Let `heldout_logMSE` be the mean squared error of
`log(C_pred)` versus the noisy held-out `log C`, and `complexity` the node
count of your expression. With `LAMBDA = 0.002`:

```
F = heldout_logMSE * (1 + LAMBDA * complexity)
B = baseline_logMSE * (1 + LAMBDA * 1)     # baseline = constant train-mean log C
Ratio = min(1000, 100 * B / F) / 1000
```
A predicted concentration `≤ 0` at a tail point is clamped to a tiny positive
value before taking the log, incurring a heavy penalty.

## Scoring
A constant prediction reproduces the baseline (`Ratio ≈ 0.1`). A single
exponential captures the average decay but extrapolates the tail with the wrong
terminal slope, scoring modestly. Correctly resolving the two-compartment
structure drives held-out error down toward the noise floor, but that
irreducible noise and the hidden rates keep even a strong recovery below `1.0`.
Simpler expressions with the same error are rewarded via the complexity term.
The per-test score is `Ratio`; the final score averages over the ladder.

## Constraints
- `test_id` in `1..10`; `n_draws` between 24 and 60 (shrinks with difficulty).
- Assay noise grows with the test id.
- Expression output ≤ 200000 bytes, single line.

## Example (worked score)
Suppose on some test your expression yields `heldout_logMSE = 0.9` with
`complexity = 12`, while the constant baseline has `baseline_logMSE = 8.0`.
Then `F = 0.9*(1+0.002*12) = 0.9216`, `B = 8.0*1.002 = 8.016`,
`Ratio = min(1000, 100*8.016/0.9216)/1000 = 0.870`. A plain constant gives
`F ≈ B` and `Ratio ≈ 0.1`.
