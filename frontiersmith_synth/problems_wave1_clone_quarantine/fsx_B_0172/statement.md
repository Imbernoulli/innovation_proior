# Berth Turnaround Scaling Law for a Container Port

## Problem
A container terminal wants a predictive **scaling law** for how long a vessel occupies
a berth. Operations logged, for many past vessel calls, the pair of drivers

* `n` — the vessel's work content (number of container moves for the call, in hundreds), and
* `c` — the number of quay cranes assigned to it,

together with the observed **berth turnaround time** `T` (hours). The measurements are
noisy, and every logged call so far has been a *small* vessel (`n` up to ~2000) served by
*few* cranes (`c` in 2..5).

The port is about to start handling **mega-vessels** — far larger workloads and larger
crane gangs than anything in the log. You must propose a single **closed-form expression**
`T = f(n, c)` that the terminal can trust to **extrapolate** into that unseen larger-scale
regime. Your expression is judged only on how well it predicts turnaround on a held-out set
of much larger `(n, c)` operating points — not on how well it memorises the training log.

## Input (stdin)
```
M
n_1 c_1 T_1
n_2 c_2 T_2
...
n_M c_M T_M
```
* `M` — number of training rows (integer).
* Each subsequent line: `n` (real), `c` (integer number of cranes), `T` (real turnaround hours).

All training rows come from the *small-vessel* regime.

## Output (stdout)
A **single line** holding one closed-form expression in the variables `n` and `c`, written
in Python syntax. Allowed:

* operators `+  -  *  /  **  %` and unary `-`,
* numeric literals and the constants `pi`, `e`,
* functions `exp, log, sqrt, sin, cos, tan, tanh, abs, pow, log2, log10`,
* the variables `n` and `c` only.

Write numeric literals as **whitespace-separated tokens** (e.g. `1.8 + 0.045 * n ** 1.15 / c ** 0.85`).
No other names, statements, function definitions, or attributes are permitted.

## Feasibility
The expression must parse under the whitelist above, use only `n` and `c` as variables, and
evaluate to a **finite real number** at every held-out point. Any violation (unknown name,
disallowed syntax, `nan`/`inf`/complex result, empty output) scores **0**.

## Objective
Minimise the relative prediction error on the hidden **held-out extrapolation split**
(operating points with `n` and `c` strictly larger than any training row), with a mild
penalty on expression size to discourage overfit constructions.

## Scoring
Let `F` be the relative RMSE of your expression on the held-out split
(`sqrt(mean(((pred-obs)/obs)^2))`) multiplied by a small complexity factor, and let `B` be
the relative RMSE of the checker's internal **constant-mean predictor**. The reported score is

```
Ratio = min(1.0, 0.1 * B / F)
```

so a constant predictor scores ~0.1 and a model that predicts ~10x better than the constant
baseline caps at 1.0. Observation noise on the held-out split leaves irreducible headroom, so
a perfect score is unattainable. The score is deterministic.

## Constraints
* `60 <= M <= 200`.
* `1 <= testId <= 10` (larger testId = more rows but noisier observations).
* Deterministic scoring; the held-out split is regenerated inside the checker with a fixed seed.

## Example (illustrative FORM only — NOT the hidden law)
Given a different dataset, a valid submission might read
```
2.0 + 5.0 / (1.0 + exp(-0.3 * (n - 800.0)))
```
This logistic shape is shown **only to demonstrate output syntax**; it is unrelated to the
scaling law behind this problem's data, which you must discover from the training rows.

### Worked score
Suppose the checker's constant baseline gives `B = 0.60` and your expression achieves
held-out relative RMSE `F = 0.12` (after the complexity factor). Then
`Ratio = min(1.0, 0.1 * 0.60 / 0.12) = min(1.0, 0.5) = 0.5`.
