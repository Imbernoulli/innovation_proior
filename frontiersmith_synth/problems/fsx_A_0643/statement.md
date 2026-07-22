# Flume Drag Crossover — recovering a hidden second mechanism from tail curvature

A tow-tank flume measures the **drag coefficient** `Cd` of a fixed sphere across
a range of **Reynolds numbers** `Re` (the dimensionless group combining fluid
density, tow velocity, sphere size and fluid viscosity). One physical
sphere/fluid combination obeys a fixed but undocumented law: `Cd` is the sum of
**two power-law mechanisms** in `Re` — a dominant one that governs the whole
range the flume can reach, and a subdominant one that is only faintly visible
there but would take over far beyond the flume's reach. Your job is to
reverse-engineer a closed-form law `Cd(Re)` that stays correct **outside** the
measured range.

The catch: the flume can only tow the sphere across a **modest** Reynolds-number
window. In that window the dominant mechanism explains almost everything —
a single best-fit power law tracks the measurements down to the noise floor.
You are graded on a **held-out extrapolation grid** reaching several decades
past the flume's window, where the subdominant mechanism has taken over and
the true curve bends far away from any single power law.

## Input (stdin)

```
t  N
Re_0  Cd_0
Re_1  Cd_1
...
Re_(N-1)  Cd_(N-1)
```

`t` is the test id. Then `N` measurement rows follow (already sorted by `Re`),
each a Reynolds number and its measured drag coefficient. The held-out grading
grid (a much wider, much higher range of `Re`) is **not** given to you.

## Output (stdout): a closed-form law

Emit a single expression for `Cd` as a function of `Re`. Allowed: numeric
constants, the operators `+ - * /`, unary `+/-`, parentheses, the single
variable `Re`, and the functions `absv(a)`, `minv(a,b)`, `maxv(a,b)`,
`powv(a,b)` (computes `a` to the power `b`; `a` must evaluate positive).

**Illustrative FORM only — NOT the hidden law:**

```
3.1 + 0.4*absv(Re - 12.0) / (1.0 + 0.02*Re)
```

This only shows the syntax; the real law's shape, exponents and coefficients
are different and must be discovered from the data.

## Feasibility

The expression must parse under the grammar above (only known names/functions,
correct arities, finite constants, at most 200 expression nodes). Any parse
violation, or any non-finite or non-positive value produced while evaluating
the law on the grading grid, scores `0` for that test.

## Objective (minimise)

Let `pred_k` be your law evaluated at held-out Reynolds number `Re_k`, and
`true_k` the (noisy) true drag coefficient there. The grader forms the mean
**squared LOG error** (it rewards matching the decay *rate*, not just the
scale) plus a small parsimony tax on expression size `nodes`:

```
F = mean_k (log(pred_k) - log(true_k))^2 * (1 + LAMBDA * nodes)
B = mean_k (log(Cbar)   - log(true_k))^2 * (1 + LAMBDA * 1)   # Cbar = flat
                                            # geometric mean of your OWN
                                            # training Cd values
Ratio = min(0.90, 0.1 * (B / F) ** GAMMA)
```

with small fixed constants `LAMBDA, GAMMA` (`0 < GAMMA < 1`), capped below 1
so the score never saturates. Predicting the flat training average reproduces
`B/F = 1` (Ratio = 0.1); a law with the right asymptotic decay rate drives
`F` down and pushes `B/F` above 1, raising the Ratio. The sub-linear exponent
`GAMMA` keeps a merely-correct-shaped law from saturating even though `B/F`
can span orders of magnitude once the decay rate is close. Measurement noise
and the finite training sample keep even a strong law below the ceiling —
report the highest Ratio you can.

## Why the flume window is a trap

Inside the flume's window the dominant mechanism is so strong that a single
power law fits the measurements beautifully — nothing in-sample signals that
anything is missing. But the subdominant mechanism decays **more slowly**, so
it leaves a small, systematically **curved** residual that grows as `Re`
approaches the top of the window. That curvature — concentrated in the
**tail** of the training sweep — is the only evidence of the hidden second
mechanism and its exponent. Fit the dominant term from the low end of the
sweep, then read the tail residual's own log-log slope to recover the rest;
one power law cannot represent two exponents and will extrapolate at the
wrong rate.

## Constraints

Time limit 5 s, memory 512 MB. `N = 90` rows; scoring is fully deterministic.
