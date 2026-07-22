# Ceiling-Regime Billiards

## Problem
In the basement of an alien physics lab, two fixed billiard balls of **unknown,
constant masses** are fired head-on at each other, over and over. Every head-on
collision is one fixed **deterministic map**: the pre-collision velocities
`(u1, u2)` become the post-collision velocities `(v1, v2)`. The lab's physics is
not ordinary — velocities can never reach a fixed **ceiling** `c`, and the map
exactly conserves two bookkeeping quantities (a momentum-like invariant and an
energy-like invariant). The two masses and the ceiling `c` are fixed for a given
logbook but are **never written down**.

The catch: the logbook only records **gentle** shots, where every speed stays
well inside the ceiling. In that regime the map looks almost like a plain linear
exchange, so its curvature — and the value of the ceiling — is barely visible.
The lab now needs to predict `v1` for **violent** shots fired close to the
ceiling. You must produce a closed-form model of `v1` that still holds in that
high-speed regime, not one that merely memorises the gentle logbook.

Each test id is a different logbook (different masses, different ceiling).

## Input (stdin)
```
line 1:          n_shots   test_id
next n_shots lines: u1 u2 v1 v2      (space-separated floats)
```
Each row is one logged gentle collision: pre-velocities `u1, u2` and **both**
post-velocities `v1, v2`. `test_id` is for reference only.

## Output (stdout)
A **single line**: a Python expression for `v1` in the variables `u1, u2`.
Allowed operators: `+ - * / ** %`; allowed functions:
`exp, log, sin, cos, sqrt, tanh, abs`; numeric literals allowed. No other names,
attributes, calls, or imports. (The held-out shots give only the pre-state, so
your expression may use `u1, u2` only — `v2` is logged history to help you infer
the physics, not an input at prediction time.)

Example output line (illustrative FORM only — **not** the hidden law):
```
0.4*u2 - 0.15*u1**2 + sqrt(abs(u1))
```

## Feasibility
The output must be exactly one line, parse under the allowed grammar, and
evaluate to a finite real number on every held-out shot. Anything else scores
`0`.

## Objective (minimise held-out error, complexity-penalised)
The grader deterministically regenerates a **held-out violent split** of
high-speed collisions (speeds pushed near the ceiling — a different, more
extreme region than the logbook) plus irreducible measurement noise, then
evaluates your expression there. Let `heldout_MSE` be the mean squared error on
`v1` and `complexity` the node count of your expression. With `LAMBDA = 0.0002`:

```
F = heldout_MSE * (1 + LAMBDA * complexity)
B = baseline_MSE * (1 + LAMBDA * 1)        # baseline = constant train-mean v1
Ratio = min(1000, 100 * B / F) / 1000
```

## Scoring
A constant prediction reproduces the baseline (`Ratio ≈ 0.1`). Driving held-out
error toward the irreducible-noise floor raises the ratio, but that floor and
the hidden ceiling keep even a strong recovery below `1.0`. A black-box fit that
models the map directly interpolates the gentle logbook yet over-shoots the
ceiling on the violent split; a model that instead pins the **conserved
quantities** (regime-independent) extrapolates correctly. Simpler expressions
with the same error score higher via the complexity term. The per-test score is
`Ratio`; the final score averages over the difficulty ladder.

## Constraints
- `test_id` in `1..10`; `n_shots` between 82 and 190 (shrinks with difficulty).
- Measurement noise grows with the test id.
- Expression output ≤ 200000 bytes, single line.

## Example (worked score)
Suppose on some test your expression gives `heldout_MSE = 0.20` with
`complexity = 60`, while the constant baseline has `baseline_MSE = 2.00`. Then
`F = 0.20*(1+0.0002*60) = 0.2024`, `B = 2.00*(1+0.0002) = 2.0004`,
`Ratio = min(1000, 100*2.0004/0.2024)/1000 = 0.988` — but realistic ceiling
noise leaves a strong recovery well under that, and a constant gives
`F ≈ B`, `Ratio ≈ 0.1`.
