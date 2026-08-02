# Calibration Burst

An unknown instrument blurs sharp events. A clean spike train `x[]` (mostly
zero, with a handful of positive pulses of unknown height) passes through
the instrument and comes out as a noisy trace

```
y[n] = sum_{j=-w}^{w}  h_w[j] * x[n-j]  +  noise[n]
```

where `h_w` is a **normalised triangular (tent) kernel** of half-width `w`:
`h_w[j] = (w+1-|j|) / (w+1)^2` for `|j| <= w`. The kernel *family* (tent
shape) is public and `w` is known to lie in `[1, 5]`, but the exact `w` for
this instrument — and the noise level — are **not told to you**; you must
read them off the data.

To help, you fired a known **calibration burst**: a test pulse train
`x_cal[]` through the same instrument, and logged both the pulses and the
blurred output `y_cal[]`. You will then be graded on a **fresh, unpaired**
trace from the same instrument (new pulses, same kernel, comparable noise)
where you only see the blurred output — never the pulses.

## Input (stdin)

```
Ncal R t
x_cal[0]  y_cal[0]
x_cal[1]  y_cal[1]
...
x_cal[Ncal-1]  y_cal[Ncal-1]
```

`R = 7` is the window radius you may reference (see below). `t` is the test
id. The held-out grading trace is NOT given to you.

## Output (stdout): one closed-form filter expression

Emit a single arithmetic expression that predicts the spike value at a query
index from a **15-tap window** of the (unpaired, held-out) blurred trace
centred on that index. The allowed variables are

```
ym7 ym6 ym5 ym4 ym3 ym2 ym1  y0  yp1 yp2 yp3 yp4 yp5 yp6 yp7
```

(`ymK` = the trace value `K` steps before the query index, `y0` = at the
query index, `ypK` = `K` steps after). Allowed operators: `+ - * /`,
parentheses, numeric constants, and the unary functions `sig` (logistic),
`step` (1 if arg>0 else 0), `relu`, `tanh`, `absv`. The expression must use
`<= 100` nodes total.

**Illustrative FORM only — NOT the hidden filter:**

```
relu( 0.4*y0 - 0.1*(ym1+yp1) ) - 0.05*tanh(ym3+yp3)
```

This only shows the syntax; the real recovery filter for this instrument
must be discovered from the calibration burst.

The grader evaluates your **single, unchanged** expression at every interior
index of the held-out trace (rolling window, no state carried between
indices) to produce a predicted spike train, and compares it to the true
held-out spikes (which you never see).

## Feasibility

The expression must parse under the grammar above (known names/functions
only, finite constants, `<=60` nodes). Any violation, or any non-finite
value produced anywhere during evaluation, scores `0`.

## Objective (maximise)

Let `MSE` be the mean squared error of your predictions against the true
held-out spikes, and `nodes` the expression's node count. The grader forms

```
F = MSE * (1 + LAMBDA * nodes)
B = MSE_of_predicting_all_zero * (1 + LAMBDA * 1)   # internal baseline
Ratio = min(1000, 100 * B / F) / 1000
```

with a small fixed `LAMBDA`. Predicting zero everywhere reproduces `B`
(Ratio ≈ 0.1). A filter that recovers more true signal than noise it injects
raises the score.

## Why sharper is not always better

Inverting a blur amplifies noise most exactly where the kernel's response is
small — pushing your filter's effective resolution past what the calibration
burst can actually support does not sharpen the recovery, it injects
amplified noise instead. A filter fit to explain the calibration burst
*perfectly* (using every available tap, unregularised) can look excellent
in-sample and still be catastrophically worse than predicting nothing once
rolled onto a fresh noise draw. The calibration burst tells you where the
kernel's real support — and the noise floor — actually are; the winning
strategy estimates that boundary and stops there rather than chasing maximum
sharpness.

## Constraints

`Ncal` up to a few hundred rows. Time limit 5 s, memory 512 MB. Scoring is
fully deterministic.
