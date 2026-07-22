# Past the Congestion Threshold

A staged production line's output metric `y` responds to a control setting
`x` (the feed rate). Every line obeys a fixed but undocumented **two-regime**
law:

- **Calm regime** (`x` below a hidden threshold `xc`): the line absorbs feed
  smoothly, `y` grows gently.
- **Cascade regime** (`x` at or above `xc`): past the threshold, backpressure
  cascades through the downstream stages and the *excess* output grows as a
  **power law of the distance past threshold**, with a hidden **scaling
  exponent** — some lines cascade mildly, some explosively.

Reverse-engineer a closed-form law `y(x)` that stays correct **far past**
where the line was ever actually run.

## Input (stdin)
```
t  N
x_0  y_0
x_1  y_1
...
x_(N-1)  y_(N-1)
```
`t` is the test id, then `N` rows follow (sorted by `x`), each a feed-rate
setting and the (noisy) output measured there. The training run covers a
**middle band**: it starts at a low feed rate and reaches only a **modest**
distance past `xc` — enough to see that *something* changes, not enough to
pin down the cascade's long-run behavior. The held-out grading grid reaches
far deeper into the cascade regime and is **never given to you**.

## Output (stdout): a closed-form law
Emit a single expression for `y` as a function of `x`. Allowed: numeric
constants, the operators `+ - * /`, unary `+/-`, parentheses, the single
variable `x`, and the functions `absv(a)` and `powv(a,b)` (computes `a` to
the power `b`; `a` must evaluate to a positive number).

**Illustrative FORM only — NOT the hidden law:**
```
2.0 + 0.5*absv(x - 3.0)
```
This only shows the syntax; the real law's threshold, exponent and
coefficients are different and must be discovered from the data.

## Feasibility
The expression must parse under the grammar above (known names/calls only,
correct arities, finite constants, at most 200 nodes). Any parse violation,
or any non-finite/non-positive value on the grading grid, scores `0`.

## Objective (maximise)
Let `pred_k` be your law evaluated at held-out feed rate `x_k`, and `true_k`
the (noisy) true output there. The grader forms the mean **squared LOG
error** (it rewards recovering the correct *growth rate*, not just matching
one scale) plus a small parsimony tax on expression size `nodes`:
```
F    = mean_k (log(pred_k) - log(true_k))^2 * (1 + LAMBDA * nodes)
base = mean_k (log(ybar)   - log(true_k))^2 * (1 + LAMBDA * 1)   # ybar =
                                                # flat geometric mean of
                                                # YOUR OWN training y values
Ratio = min(CAP, 0.1 * (base / F) ** GAMMA)
```
with small fixed constants `LAMBDA, GAMMA > 0` and a hard cap `CAP < 1`.
Predicting the flat training average everywhere reproduces `base/F = 1`
(`Ratio = 0.1`); the right long-run growth rate drives `F` down and pushes
the ratio up. The sub-linear `GAMMA` keeps a merely-right-shaped law from
saturating even though `base/F` can span orders of magnitude once the growth
rate is close (the grading grid reaches several decades past the training
band). Noise and the finite sample keep even a correct law below the cap.

## Why the obvious fit is a trap
The training band looks like ordinary scaling data — smoothly increasing,
mildly bending upward — so the obvious move is a single global power-law
fit, `y = C*x^p`, by log-log least squares over every row. It tracks the
training band decently (both regimes are monotone increasing), because
nothing in-sample screams "two mechanisms." But it silently assumes the
power law runs through the origin, when the cascade only turns on at `xc`.
That misattributes the calm regime's own curvature into the fitted exponent,
which then does not match the true cascade exponent — extrapolated far past
the training band, this recipe grows at the wrong rate.

The insight is to **search for the regime boundary** under which a
transformed variable — the residual above the boundary, in log-log space
against the *distance past the boundary* — collapses onto one clean line.
Locating that boundary and reading its slope recovers the true exponent
directly, and because it is the real functional form (not a local patch), it
extrapolates correctly arbitrarily far into the cascade regime.

## Constraints
Time limit 5 s, memory 512 MB. `N = 90` rows; scoring is fully
deterministic.
