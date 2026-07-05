# Real-Gas Equation of State: Extrapolate a Cylinder EOS into the High-Pressure Regime

## Problem
In a gas-cylinder calibration lab, a pure real gas is charged into cylinders of
known volume. For a grid of molar volumes `V` (litres per mole) and temperatures
`T` (kelvin) the lab logs the measured pressure `P` (bar). Every logged pressure
carries a small **relative** measurement error.

The lab only ever samples the **dilute-to-moderate** regime (molar volumes down
to `0.22 L/mol`). Safety engineering, however, needs the pressure in the
**high-pressure regime** — small molar volumes `V` in `[0.11, 0.18] L/mol`, where
the cylinders are never actually filled during calibration.

Recover a single **closed-form equation of state** `P(V, T)` from the training
log that **extrapolates** accurately into that unsampled high-pressure region.
There is genuine irreducible measurement noise, and no exact fit is achievable —
better functional structure generalizes better.

## Input (stdin)
Many lines, one measurement per line:

```
<V> <T> <P>
```

`V` is the molar volume (L/mol), `T` the temperature (K), `P` the measured
pressure (bar). All training rows have `V >= 0.22`.

## Output (stdout)
A **single closed-form expression** for the pressure, written as a Python
expression string in the variables `V` and `T`. The last non-empty line of your
output is taken as the expression (a leading `P = ` is tolerated and stripped).

Allowed tokens: the variables `V` and `T`; numeric constants; the operators
`+ - * / ** %`; the functions `log, log10, exp, sqrt, sin, cos, tanh, abs, pow`;
and the constants `pi, e`. Any other name/construct is rejected.

Example of the required OUTPUT FORMAT (this is an **illustrative FORM only — it
is NOT the hidden law**, only a demonstration of syntax):

```
P = 0.05*T + 3.0/V - 1.2*sqrt(T)/V
```

## Feasibility
The output must parse under the whitelist above, use only `V`, `T` and allowed
tokens, and evaluate to a **finite** real number at every held-out point.
Non-finite (`nan`/`inf`), out-of-vocabulary, unparseable, absurdly large
(`|P| > 1e7`), or empty outputs score `0`.

## Objective (minimize)
Let `err` be the root-mean-square error of your expression against the true
pressures on the hidden **high-pressure held-out grid** (small `V`, regenerated
deterministically inside the checker), with a mild penalty for expression size.
The objective to minimize is this held-out error. Lower is better.

## Scoring
The checker builds an internal baseline `B` = the held-out RMSE of a two-term
virial fit `P = k*T/V + c/V**2` least-squares-fitted to your training log. With
`eff` = your penalized held-out RMSE:

```
sc    = min(1000, 100 * B / max(1e-9, eff))
Ratio = sc / 1000        # printed on the last line
```

Reproducing the virial baseline scores `Ratio ~= 0.1`. Capturing the
high-pressure physics (the excluded-volume pressure pole and the
temperature-dependent attraction) drives the held-out error well below the
baseline and lifts the score. Irreducible noise keeps a perfect score out of
reach.

## Constraints
- 10 test cases (a difficulty ladder: later cases carry more measurement noise).
- Deterministic scoring; seed-stable; O(size).
- Held-out region is strictly higher pressure (smaller `V`) than any training row.

## Example (worked score)
Suppose on some case the virial baseline yields held-out RMSE `B = 32.0` bar and
your submitted expression yields penalized held-out RMSE `eff = 8.0` bar. Then
`sc = min(1000, 100 * 32.0 / 8.0) = 400`, so `Ratio = 0.400`. If instead your
expression evaluated to `inf` at any held-out point, the output is infeasible and
`Ratio = 0.0`.
