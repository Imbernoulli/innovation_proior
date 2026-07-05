# Arrhenius Rate Law for a Catalytic Packed-Bed Reactor

## Problem
A catalysis group runs a fixed-bed reactor and logs the intrinsic **reaction-rate
constant** `k` (mol per litre per second) of a surface-catalysed conversion as a
function of two controllable drivers:

* `T` — the absolute bed **temperature** in kelvin, and
* `C` — the reactant **concentration** at the catalyst face in mol per litre.

Every entry in the log so far was measured on a bench rig that can only run **cold and
dilute**: temperatures were held in a low band and concentrations were kept modest.
The measurements are noisy (calibration drift, sampling error).

The group is about to commission a full-scale unit that will run **much hotter and much
more concentrated** than anything on the bench. You must propose a single **closed-form
expression** `k = f(T, C)` — a candidate *rate law* — that the engineers can trust to
**extrapolate** into that unseen hot, concentrated regime. Your expression is graded only
on how well it predicts the rate on a held-out set of much larger `(T, C)` operating
points, **not** on how tightly it memorises the cold bench log.

## Input (stdin)
```
M
T_1 C_1 k_1
T_2 C_2 k_2
...
T_M C_M k_M
```
* `M` — number of training rows (integer).
* Each subsequent line: `T` (real kelvin), `C` (real mol/L), `k` (real rate constant).

All training rows come from the *cold, dilute* bench regime.

## Output (stdout)
A **single line** holding one closed-form expression in the variables `T` and `C`, written
in Python syntax. Allowed:

* operators `+  -  *  /  **  %` and unary `-`,
* numeric literals and the constants `pi`, `e`,
* functions `exp, log, sqrt, sin, cos, tan, tanh, abs, pow, log2, log10`,
* the variables `T` and `C` only.

Write numeric literals as **whitespace-separated tokens**
(e.g. `1000000.0 * exp(-6500.0 / T) * C ** 0.7`).
No other names, statements, function definitions, comprehensions, or attribute access are
permitted.

## Feasibility
The expression must parse under the whitelist above, use only `T` and `C` as variables, and
evaluate to a **finite real number** at every held-out point. Any violation (unknown name,
disallowed syntax, `nan`/`inf`/complex result, empty output) scores **0**.

## Objective
**Minimise** the relative prediction error on the hidden **held-out extrapolation split**
(hot, concentrated operating points with `T` and `C` strictly larger than any training row),
with a mild penalty on expression size to discourage overfit constructions.

## Scoring
Let `F` be the relative RMSE of your expression on the held-out split
(`sqrt(mean(((pred-obs)/obs)^2))`) multiplied by a small complexity factor, and let `B` be
the relative RMSE of the checker's internal **constant-mean predictor** (it predicts the mean
of the training `k` values everywhere). The reported score is

```
Ratio = min(1.0, 0.1 * B / F)
```

so the constant predictor scores ~0.1 and a model predicting ~10x better than that baseline
caps at 1.0. Observation noise on the held-out split plus the long extrapolation leave
irreducible headroom, so a perfect score is unattainable. The score is deterministic.

## Constraints
* `72 <= M <= 180`.
* `1 <= testId <= 10` (larger testId = more rows but noisier bench measurements).
* Deterministic scoring; the held-out split is regenerated inside the checker with a fixed seed.

## Example (illustrative FORM only — NOT the hidden law)
Given a different dataset, a valid submission might read
```
0.5 + 12.0 * tanh(0.001 * T) * sqrt(C)
```
This shape is shown **only to demonstrate output syntax**; it is unrelated to the rate law
behind this problem's data, which you must discover from the training rows.

### Worked score
Suppose the checker's constant baseline gives `B = 0.95` and your expression achieves
held-out relative RMSE `F = 0.19` (after the complexity factor). Then
`Ratio = min(1.0, 0.1 * 0.95 / 0.19) = min(1.0, 0.5) = 0.5`.
