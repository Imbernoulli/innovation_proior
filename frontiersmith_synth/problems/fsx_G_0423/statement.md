# Outbreak Incidence Law Recovery — extrapolating the post-peak tail

## Problem
You run the analytics behind a public-health **outbreak dashboard**. A new
outbreak is unfolding and, week by week, the dashboard logs the reported new-case
**incidence** `y` together with four normalised drivers:

| variable | meaning |
|----------|---------|
| `x1` | time `t` in weeks since first detection |
| `x2` | non-pharmaceutical-intervention **stringency** in `[0,1]` |
| `x3` | surveillance / **testing** intensity in `[0,1]` |
| `x4` | a calendar/mobility **nuisance** signal in `[0,1]` |

There is a fixed but hidden **incidence law** `y ≈ f(x1,x2,x3,x4)`: a single-peak
epidemic wave (a fast rise, a slower decay) modulated by the surveillance
drivers, observed through measurement noise. Your job is to **recover a
closed-form incidence law** from the early-outbreak reports so the dashboard can
project the outbreak forward.

The catch: the dashboard only exposes the **early window** — the rise, the peak,
and the very start of the decline (`x1 ∈ [0, 11]`). You are graded on the
**post-peak tail** (`x1 ∈ [11, 17.5]`), a time region your training data never
covers. This is genuine extrapolation: you must infer the *decay law* from data
that mostly shows the *rise*.

## Input (stdin)
```
n t
x1 x2 x3 x4 y      (row 1)
...                (n rows total)
```
The first token `n` is the number of training weeks; `t` is the test id. Each of
the next `n` lines is one noisy reported observation. The hidden law, its seed and
its coefficients are never given.

## Output (stdout)
A **single line**: one closed-form Python expression in the variables
`x1, x2, x3, x4`. Allowed operators `+ - * / ** %` and allowed functions
`exp, log, sin, cos, sqrt, tanh, abs`. No other names, calls, attributes or
imports. Example of the *required output shape* (this is an **illustrative FORM
only — NOT the hidden law**):

```
3.0*x1 + sqrt(abs(x2)) - 0.5*x3*x4 + 1.2
```

## Feasibility
The expression must parse under the whitelist above and evaluate to a **finite
real number** at every held-out point. Any parse error, disallowed name/call,
non-numeric or non-finite (`nan`/`inf`) result scores `0`.

## Objective (minimize)
Minimise the complexity-penalised **held-out mean squared error** on the
post-peak-tail split. Let `M` be the MSE of your expression on the held-out
points and `c` the expression complexity (node count). The grader forms
`F = M · (1 + λ·c)` and compares it to the constant-mean baseline
`B = B_mse · (1 + λ·1)`.

## Scoring
Deterministic. The grader regenerates the hidden law, the training sample and the
held-out post-peak-tail split from `t` alone, evaluates your expression there, and
prints
```
Ratio = min(1000, 100 · B / F) / 1000
```
A constant equal to the training mean reproduces the baseline (`Ratio ≈ 0.1`).
Recovering the wave shape — especially its slow exponential tail — drives `F`
down and the ratio up. An **irreducible noise floor** injected into the tail plus
the fact that the rise/decay rates are only glimpsed before the tail keep the
score well below `1.0`; there is no reachable perfect law.

## Constraints
- `1 ≤ t ≤ 10`; `76 ≤ n ≤ 220`.
- Output at most one non-empty line; ≤ 200000 bytes.
- Higher test ids = more measurement noise + fewer reported weeks.

## Example (worked score)
Suppose your line is the constant `12.7` (the training mean). Then `F = B` and
`Ratio = 100/1000·(1+λ) / (1+λ) = 0.1`. If instead you submit a mechanistic wave
whose tail matches the true decay, the held-out MSE drops several-fold and the
ratio climbs toward — but stays under — the noise-limited ceiling.
