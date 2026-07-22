# Six Miscalibrated Labs, One Law

Five independent laboratories each built an apparatus to measure the same
underlying physical quantity as a function of a control parameter `x`. Every
lab's readout, however, passes through its own uncalibrated amplifier and has
its own zero-drift: lab `r` reports

```
y = gain_r * core(x) + offset_r + noise
```

where `core(x)` is the ONE shared law common to all labs, and `gain_r`,
`offset_r` are unknown, lab-specific nuisance constants you are never told.
Worse, each lab's apparatus only operates over a narrow, lab-specific window
of `x` — no two labs' windows overlap, and together the five windows still
leave large gaps uncovered.

Your job: recover `core(x)` well enough that it correctly predicts **brand-new
labs** — ones you never saw during training, each with its own unknown
gain/offset — operating in an `x` range that lies completely outside every
window you were shown.

## Input (stdin)

```
K t
r  n  lo  hi
x_0 y_0
...
x_{n-1} y_{n-1}
(repeated for each of the K regimes)
```

`t` is the test id. `K` regimes follow; regime `r`'s window is `[lo, hi]`,
followed by its `n` noisy `(x, y)` rows (floats).

## Output (stdout): one closed-form expression for `core(x)`

Print a single arithmetic expression in the variable `x`, using `+ - * /`,
parentheses, numeric constants, and the unary functions `sin`, `cos`, `exp`,
`abs`. At most 60 expression nodes.

**Illustrative FORM only — NOT the hidden law:**
```
0.4 * exp(0.1 * x) - abs(x)
```
This just shows the syntax; the real shared law has a different shape and you
must discover it from the data.

## Feasibility

The expression must parse under the grammar above (known names/functions
only, finite constants, node count within bounds) and must evaluate to a
finite number everywhere it is queried. Any violation scores `0`.

## Scoring (minimise)

The grader never reveals the held-out labs to you. Internally, for each of
several fresh, never-shown labs it:

1. Draws a handful of **calibration** points `(x, y)` from that lab's own
   narrow window.
2. Fits the best affine rescaling `y ≈ gain * core_hat(x) + offset` of your
   submitted `core_hat` against those calibration points only (ordinary least
   squares in `gain, offset`) — this is how the grader removes ITS OWN
   nuisance uncertainty about the new lab, the same trick you should use to
   remove nuisance uncertainty across the labs you were shown.
3. Uses that fitted `gain, offset` to predict `y` at a fresh set of
   **extrapolation** points, drawn from an `x` range outside every window
   (training or calibration) you have ever seen.

It then pools the squared errors from every held-out lab into one mean
squared error `F`, against an internal baseline `B` = the pooled MSE of
simply predicting each lab's own calibration mean everywhere (i.e. the score
`core_hat(x)=0` would get). Prints

```
Ratio = min(1000, 100 * B / F) / 1000
```

A flat guess reproduces `B` (Ratio ≈ 0.1). The better `core_hat` matches the
TRUE shared law's shape (not just any curve that happens to fit one lab's
narrow window), the lower `F` and the higher the Ratio — but sensor noise
keeps even a very good recovery below the ceiling.

## Why per-lab curve fitting is a trap

Every lab's window is narrow, so a flexible local fit (say, a cubic in `x`,
or a sinusoid at almost any frequency) can match one lab's rows to near-zero
residual — the window simply isn't wide enough to distinguish the true shape
from many impostors. That near-perfect local fit carries no information
about how the curve behaves in the gaps between labs, let alone far beyond
every window where the sixth lab lives; a shape chosen to satisfy one lab in
isolation typically diverges badly once extrapolated there. The nuisance
`gain_r, offset_r` also confounds naive pooling: stacking every lab's raw
rows together and fitting one curve treats each lab's arbitrary vertical
shift and scale as if it were signal about `x`.

## Constraints

Time limit 5 s, memory 512 MB. `K = 5`, each window has 34–38 rows. Scoring
is fully deterministic.
