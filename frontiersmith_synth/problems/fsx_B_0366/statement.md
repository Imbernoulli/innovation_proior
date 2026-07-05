# Life-Support Power Scaling Law for a Lunar Habitat

## Problem
An agency operates a growing cluster of pressurized **lunar habitats**. For each habitat it
logs two design drivers,

* `k` — the crew size it sustains (number of astronauts), and
* `V` — the pressurized internal volume it encloses (cubic metres),

together with the observed steady-state **life-support electrical power** `P` it draws
(kilowatts) to run air revitalisation, thermal control, water recovery and lighting. The
telemetry is noisy, and every habitat commissioned so far is **small**: crews of only a
handful of people (`k` up to ~12) inside modest modules (`V` up to ~300 m^3).

The agency is now designing **large surface bases** — big pressurized volumes housing large
crews, far beyond anything flown. You must propose a single **closed-form expression**
`P = f(k, V)` that mission planners can trust to **extrapolate** into that unseen large-scale
regime. Your expression is judged only on how well it predicts power draw on a held-out set
of much larger `(k, V)` design points — not on how well it memorises the training telemetry.

## Input (stdin)
```
M
k_1 V_1 P_1
k_2 V_2 P_2
...
k_M V_M P_M
```
* `M` — number of training rows (integer).
* Each subsequent line: `k` (integer crew), `V` (real volume, m^3), `P` (real power, kW).

All training rows come from the *small-habitat* regime.

## Output (stdout)
A **single line** holding one closed-form expression in the variables `k` and `V`, written
in Python syntax. Allowed:

* operators `+  -  *  /  **  %` and unary `-`,
* numeric literals and the constants `pi`, `e`,
* functions `exp, log, sqrt, sin, cos, tan, tanh, abs, pow, log2, log10`,
* the variables `k` and `V` only.

Write numeric literals as **whitespace-separated tokens** (e.g. `3.0 + 0.02 * k ** 1.2 * V ** 0.6`).
No other names, statements, function definitions, or attributes are permitted.

## Feasibility
The expression must parse under the whitelist above, use only `k` and `V` as variables, and
evaluate to a **finite real number** at every held-out point. Any violation (unknown name,
disallowed syntax, `nan`/`inf`/complex result, empty output) scores **0**.

## Objective
Minimise the relative prediction error on the hidden **held-out extrapolation split**
(design points with `k` and `V` strictly larger than any training row), with a mild penalty
on expression size to discourage overfit constructions.

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
* `70 <= M <= 210`.
* `1 <= testId <= 10` (larger testId = more rows but noisier telemetry).
* Deterministic scoring; the held-out split is regenerated inside the checker with a fixed seed.

## Example (illustrative FORM only — NOT the hidden law)
Given a different dataset, a valid submission might read
```
1.5 + 4.0 * tanh(0.001 * V) + 0.3 * k
```
This shape is shown **only to demonstrate output syntax**; it is unrelated to the scaling law
behind this problem's data, which you must discover from the training rows.

### Worked score
Suppose the checker's constant baseline gives `B = 0.85` and your expression achieves held-out
relative RMSE `F = 0.17` (after the complexity factor). Then
`Ratio = min(1.0, 0.1 * 0.85 / 0.17) = min(1.0, 0.5) = 0.5`.
