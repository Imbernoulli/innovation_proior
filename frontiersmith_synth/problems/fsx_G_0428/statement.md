# Integer-Sequence Oracle — extrapolating recurrence discovery

## Problem
A combinatorics lab exposes an **oracle** that emits the terms of a hidden
integer sequence `a(0), a(1), a(2), ...`. Under the hood each term is a
**combinatorial count** governed by a fixed but **unknown** law — an integer
linear recurrence (think Fibonacci / `k`-nacci growth, or a figurate
polynomial). Because the counts are produced by a *stochastic sampler*, every
reported term carries a small bounded **multiplicative jitter**: what you read
is the true count nudged by an unpredictable few-percent error.

You are handed only the **first `T` reported terms**. The lab now needs the
sequence's behaviour in a **far-future window** — indices well beyond anything
you have seen — where the oracle will report **freshly jittered** terms. Your
job is to propose a compact **closed-form or recurrence** for `a(n)` that
**generalises to that far-future window**, rather than one that merely memorises
the noisy prefix.

Each test id corresponds to a different hidden sequence (a different law).

## Input (stdin)
```
line 1:        T   test_id
next T lines:  n   a(n)          (index and reported term, whitespace-separated)
```
`test_id` is provided for reference only; the law must be inferred from the
data. Indices run `n = 0, 1, ..., T-1`.

## Output (stdout)
A **single line** holding a Python expression for `a(n)` in the variables:

- `n`  — the integer index of the term being predicted;
- `a1, a2, a3` — the previous terms `a(n-1), a(n-2), a(n-3)` (use these to write
  a **recurrence**; ignore them to write a **closed form in `n`**).

Allowed operators: `+ - * / ** %`; allowed functions:
`exp, log, sin, cos, sqrt, tanh, abs`; numeric literals are allowed. No other
names, attributes, calls, or imports are permitted.

Example output line (illustrative **FORM only — not** the hidden law):
```
1.5 * a1 - 0.5 * a2 + 0.01 * n**2
```

## Feasibility
The output must be exactly one line, parse as an expression over the allowed
grammar, and evaluate to a **finite** real number on every held-out point.
Anything else (empty, multi-line, unknown name, import, `nan`/`inf`, evaluation
error) scores `0`.

## Objective (minimise relative held-out error, complexity-penalised)
The grader deterministically regenerates the hidden law and a **far-future
held-out window** with fresh, independent jitter. For each held-out index `n`
it supplies the true previous terms as `a1, a2, a3` (**teacher forcing** — your
prediction of one term is never fed back into the next), evaluates your
expression, and measures the relative error

```
err(n) = |pred(n) - a(n)| / (1 + |a(n)|)
```

Let `heldout_err` be the mean of `err(n)` and `complexity` the node count of
your expression. With `LAMBDA = 0.002`:

```
F = heldout_err * (1 + LAMBDA * complexity)
B = baseline_err * (1 + LAMBDA * 1)     # baseline = constant last observed term
Ratio = min(1000, 100 * B / F) / 1000
```

## Scoring
Predicting a single constant (the last observed term) reproduces the baseline
(`Ratio ≈ 0.1`). Recovering the underlying law drives `heldout_err` down toward
the **irreducible jitter floor**, raising the ratio — but that floor (fresh
far-future jitter you cannot observe) plus the fact that higher-order laws are
not exactly representable keep even a strong recovery **below `1.0`**. Simpler
expressions with the same error are rewarded via the complexity term. The
per-test score is `Ratio`; the final score averages over the difficulty ladder.

## Constraints
- `test_id` in `1..10`; `T` between 16 and 24 (shrinks with difficulty).
- Recurrence order, jitter amplitude, and extrapolation distance grow with the
  test id.
- Expression output ≤ 200000 bytes, single line.

## Example (worked score)
Suppose on some test your expression yields `heldout_err = 0.09` with
`complexity = 11`, while the constant baseline has `baseline_err = 0.82`. Then
`F = 0.09*(1+0.002*11) = 0.0920`, `B = 0.82*(1.002) = 0.8216`,
`Ratio = min(1000, 100*0.8216/0.0920)/1000 = 0.893`. A plain constant would give
`F ≈ B` and `Ratio ≈ 0.1`.
