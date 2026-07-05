# Far-Wing Implied-Volatility Surface Recovery

## Problem
An options desk quotes an equity-index book whose **implied volatility**
`sigma` follows a fixed but **unknown** closed-form surface over two variables:

- `k` — log-moneyness `= log(K / F)` (K = strike, F = forward)
- `t` — tenor in years

During the session the desk only trades a **liquid near-the-money band**
`k ∈ [-0.15, 0.15]`, `t ∈ [0.10, 2.00]`, and every quote carries a small
bid/ask microstructure error. Risk needs a compact analytic surface that still
prices the **far-strike wings** — deep out-of / in-the-money options whose
`|k|` sits **well beyond** the traded band, where smile convexity dominates.
Your job is to recover a closed-form expression for `sigma(k, t)` that
**generalises into those wings**, not one that merely memorises the noisy
near-ATM quotes.

Each test id corresponds to a different book (a different hidden surface).

## Input (stdin)
```
line 1:            n_train   test_id
next n_train lines: k t sigma        (space-separated floats)
```
`test_id` is provided for reference only; the surface must be inferred from
the data.

## Output (stdout)
A **single line** holding a Python expression for `sigma` in the variables
`k, t`. Allowed operators: `+ - * / ** %`; allowed functions:
`exp, log, sin, cos, sqrt, tanh, abs`; numeric literals are allowed. No other
names, attributes, calls, or imports are permitted.

Example output line (illustrative FORM only — **not** the hidden law):
```
0.2 + 0.1*t - 0.3*k + sqrt(abs(k))
```

## Feasibility
The output must be exactly one line, parse as an expression over the allowed
grammar, and evaluate to a finite real number on every held-out point.
Anything else scores `0`.

## Objective (minimise held-out error, complexity-penalised)
The grader deterministically regenerates a **held-out far-wing split** at
strikes with `|k|` far outside the training band, plus irreducible quote noise,
then evaluates your expression there. Let `heldout_MSE` be the mean squared
error and `complexity` the node count of your expression. With
`LAMBDA = 0.003`:

```
F = heldout_MSE * (1 + LAMBDA * complexity)
B = baseline_MSE * (1 + LAMBDA * 1)        # baseline = constant train mean
Ratio = min(1000, 100 * B / F) / 1000
```

## Scoring
A flat-vol constant reproduces the baseline (`Ratio ≈ 0.1`). Recovering the
wing convexity (the `k²` smile and its maturity decay) drives held-out error
down and raises the ratio, but the irreducible quote-noise floor and the hidden
smile-decay rate keep even a strong recovery below `1.0`. Simpler expressions
with the same error are rewarded via the complexity term. The per-test score is
`Ratio`; the final score averages over the difficulty ladder.

## Constraints
- `test_id` in `1..10`; `n_train` between 74 and 200 (shrinks with difficulty).
- Quote noise grows with the test id.
- Expression output ≤ 200000 bytes, single line.

## Example (worked score)
Suppose on some test your expression yields `heldout_MSE = 0.0009` with
`complexity = 20`, while the flat baseline has `baseline_MSE = 0.0100`. Then
`F = 0.0009*(1+0.003*20) = 0.0009540`, `B = 0.0100*1.003 = 0.010030`,
`Ratio = min(1000, 100*0.010030/0.0009540)/1000 = 1.000` (capped). A plain
constant would give `F ≈ B` and `Ratio ≈ 0.1`.
