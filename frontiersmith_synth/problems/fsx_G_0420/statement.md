# Factory CES Production-Function Recovery

## Problem
A manufacturing plant's output `y` (units produced per shift, normalised) is
governed by a fixed but **unknown** closed-form production law over two
normalised inputs:

- `K` — capital deployed (machine-hours)
- `L` — labour deployed (worker-hours)

The plant's engineers can only meter the plant during **normal operation**,
where both inputs stay inside a bounded calibration box `K, L ∈ [0.40, 1.30]`,
and every output reading is corrupted by multiplicative productivity shocks
(log-normal measurement noise). Management is planning a **capacity expansion**
and needs a compact analytic model that still predicts output correctly in the
**high-throughput regime**, where both `K` and `L` are pushed **beyond** the
calibration box. Your job is to recover a closed-form expression for `y` that
**generalises to that extrapolation region**, not one that merely memorises the
noisy training points.

Each test id corresponds to a different plant (a different hidden law with a
different capital share, returns-to-scale, and — crucially — a different, hidden
elasticity of substitution between capital and labour).

## Input (stdin)
```
line 1:            n_train   test_id
next n_train lines: K L y      (space-separated floats)
```
`test_id` is provided for reference only; the law must be inferred from the data.

## Output (stdout)
A **single line** holding a Python expression for `y` in the variables `K, L`.
Allowed operators: `+ - * / ** %`; allowed functions:
`exp, log, sin, cos, sqrt, tanh, abs`; numeric literals are allowed. No other
names, attributes, calls, or imports are permitted.

Example output line (illustrative FORM only — **not** the hidden law):
```
1.1 * K**0.5 * L**0.5 + 0.2*sin(K)
```

## Feasibility
The output must be exactly one line, parse as an expression over the allowed
grammar, and evaluate to a finite real number on every held-out point.
Anything else scores `0`.

## Objective (minimise held-out error, complexity-penalised)
The grader deterministically regenerates a **held-out extrapolation split** in a
higher-throughput region (a different, larger input box than the training data)
plus irreducible measurement noise, then evaluates your expression there. Let
`heldout_MSE` be the mean squared error and `complexity` the node count of your
expression. With `LAMBDA = 0.003`:

```
F = heldout_MSE * (1 + LAMBDA * complexity)
B = baseline_MSE * (1 + LAMBDA * 1)        # baseline = constant train mean
Ratio = min(1000, 100 * B / F) / 1000
```

## Scoring
A constant prediction reproduces the baseline (`Ratio ≈ 0.1`). Recovering the
production shape drives held-out error down and raises the ratio, but two effects
keep even a strong recovery below `1.0`: an irreducible productivity-noise floor,
and the fact that any log-polynomial (translog) surrogate is only a Taylor
approximation of the true constant-elasticity-of-substitution law and drifts in
the extrapolation region. Simpler expressions with the same error are rewarded
via the complexity term. The per-test score is `Ratio`; the final score averages
over the difficulty ladder.

## Constraints
- `test_id` in `1..10`; `n_train` between 85 and 220 (shrinks with difficulty).
- Noise grows with the test id.
- Expression output ≤ 200000 bytes, single line.

## Example (worked score)
Suppose on some test your expression yields `heldout_MSE = 0.9` with
`complexity = 24`, while the constant baseline has `baseline_MSE = 9.0`. Then
`F = 0.9*(1+0.003*24) = 0.9648`, `B = 9.0*1.003 = 9.027`,
`Ratio = min(1000, 100*9.027/0.9648)/1000 = 0.936`. A plain constant would give
`F ≈ B` and `Ratio ≈ 0.1`.
